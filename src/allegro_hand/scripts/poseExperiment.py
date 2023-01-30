#!/usr/bin/env python3
import os
import time

import cv2
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
from processData import DynamicsModel
from sensor_msgs.msg import Image, JointState
import threading

from digit.pose_estimation import Pose

np.random.seed(int(((time.time() % 1000.0) * 1000000) % (2**31)))

JointStateTopic = "/allegroHand_0/joint_states"
JointCommandTopic = "/allegroHand_0/joint_cmd"
DigitTopic0 = "digit_sensor/0/raw"
digitImage0 = np.zeros((320, 240), dtype=np.uint8)
DigitPoseTopic0 = "digit_sensor/0/pose"
digitPoseImage0 = np.zeros((320, 240), dtype=np.uint8)
currentJointPose = None
currentJointTorque = None
InitialGraspPose = [
    0.32,
    1.65,
    0.109,
    0.064,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.05,
    0.1,
    0.3,
    0.1,
]
HomePose = [
    0.0,
    -0.17453292519,
    0.78539816339,
    0.78539816339,
    0.0,
    -0.17453292519,
    0.78539816339,
    0.78539816339,
    0.08726646259,
    -0.08726646259,
    0.8726646259,
    0.78539816339,
    1.0471975512,
    0.43633231299,
    0.26179938779,
    0.78539816339,
]
bridge = CvBridge()
newPose = np.array(InitialGraspPose)
lock = threading.Lock()


class EnsembleDynamicsModel:
    def __init__(self, model_names):
        self.models = []
        for model_name in model_names:
            model = torch.load(os.path.join("data", model_name))
            model.eval()
            model = model.cuda()
            self.models.append(model)

    def __call__(self, hand_state, tactile_state, actions):
        hand_state = torch.from_numpy(hand_state).float().cuda()
        tactile_state = torch.from_numpy(tactile_state).float().cuda()
        actions = torch.from_numpy(actions).float().cuda()
        hand_state = torch.tile(hand_state, (actions.shape[0], 1))
        tactile_state = torch.tile(tactile_state, (actions.shape[0], 1))
        outputs = []
        with torch.no_grad():
            for model in self.models:
                outputs.append(
                    model.predict(hand_state, tactile_state, actions).cpu().numpy()
                    * model.contactNorm.numpy()
                )
        outputs = np.stack(outputs, axis=0)
        return outputs

    def evolveActions(
        self, hand_state, tactile_state, actions, target, optimize_steps=100, lr=0.0005
    ):
        new_hand_state = torch.from_numpy(hand_state).float().cuda()
        new_tactile_state = torch.from_numpy(tactile_state).float().cuda()
        new_actions = torch.from_numpy(actions).float().cuda()
        new_actions.requires_grad = True
        new_hand_state = torch.tile(new_hand_state, (actions.shape[0], 1))
        new_tactile_state = torch.tile(new_tactile_state, (actions.shape[0], 1))
        target = (
            torch.from_numpy(target.reshape((1, -1))).float().cuda()
        )  # target should not contain contact strength
        optimizer = torch.optim.Adam([new_actions], lr=lr)
        loss = None
        for _ in range(optimize_steps):
            outputs = []
            optimizer.zero_grad()
            for model in self.models:
                output = (
                    model(new_hand_state, new_tactile_state, new_actions)[:, 1:3]
                    - target[:, :2]
                )
                # output[:, 2] = 0.5 - torch.abs(output[:, 2] - 0.5)
                outputs.append(output)
            loss = torch.stack(outputs, dim=0)
            loss[:, 0] *= 240
            loss[:, 1] *= 320
            loss = torch.sqrt(torch.square(loss).sum(dim=1)).mean()
            loss.backward()
            optimizer.step()
        print("Optimizing....")
        new_actions = new_actions.detach().cpu().numpy()
        return new_actions, self(hand_state, tactile_state, new_actions)


def jointStateCallback(msg):
    global currentJointPose, currentJointTorque, thumbFK, indexFK, lock
    lock.acquire()
    currentJointPose = msg.position
    currentJointTorque = msg.effort
    lock.release()


def DigitCallback0(msg):
    global digitImage0, lock
    lock.acquire()
    digitImage0 = bridge.imgmsg_to_cv2(msg, "passthrough")
    lock.release()


def DigitPoseCallback0(msg):
    global digitPoseImage0, lock
    lock.acquire()
    digitPoseImage0 = bridge.imgmsg_to_cv2(msg, "passthrough")
    lock.release()


if __name__ == "__main__":

    model_names = [
        "noTdynamics_model0.pth",
        "noTdynamics_model1.pth",
        "noTdynamics_model2.pth",
        "noTdynamics_model3.pth",
        "noTdynamics_model4.pth",
        "noTdynamics_model5.pth",
        "noTdynamics_model6.pth",
        "noTdynamics_model7.pth",
        "noTdynamics_model8.pth",
        "noTdynamics_model9.pth",
        "noTdynamics_model10.pth",
        "noTdynamics_model11.pth",
        "noTdynamics_model12.pth",
        "noTdynamics_model13.pth",
        "noTdynamics_model14.pth",
        "noTdynamics_model15.pth",
    ]
    actions = np.load("actions.npy")
    actions = np.delete(actions, 4, axis=1)
    ensemble_dynamics_model = EnsembleDynamicsModel(model_names)

    rospy.init_node("controller")
    rospy.loginfo("Running controller...")
    rospy.loginfo("Press Ctrl+C to stop.")

    jointStateSub = rospy.Subscriber(JointStateTopic, JointState, jointStateCallback)
    DigitSub0 = rospy.Subscriber(DigitTopic0, Image, DigitCallback0)
    DigitPoseSub0 = rospy.Subscriber(DigitPoseTopic0, Image, DigitPoseCallback0)
    jointCommandPub = rospy.Publisher(JointCommandTopic, JointState, queue_size=1)
    rospy.sleep(1.0)

    jointStateMsg = JointState()
    pose_estimator = Pose()
    referenceImage = digitImage0.copy().astype(np.float32)
    image_shape = referenceImage.shape
    actionPerRun = 5
    runs = 30
    runId = 21
    withGradientOpt = False
    errors = []
    breakFlag = False
    while True:
        cv2.imshow("Pose image", digitPoseImage0)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("g"):
            jointStateMsg.header.stamp = rospy.Time.now()
            jointStateMsg.position = InitialGraspPose
            newPose = np.array(InitialGraspPose)
            jointCommandPub.publish(jointStateMsg)
        elif key == ord("h"):
            jointStateMsg.header.stamp = rospy.Time.now()
            jointStateMsg.position = HomePose
            jointCommandPub.publish(jointStateMsg)
            rospy.sleep(1.0)
            referenceImage = digitImage0.copy()
            save_count = 0
            print("Finished home pose..............................")
        elif key == ord("j"):
            jointStateMsg.header.stamp = rospy.Time.now()
            sample = actions[np.random.randint(0, actions.shape[0]), :]
            print("sample is ", sample)
            newPose[:4] = sample[:4].flatten()
            newPose[-3:] = sample[-3:].flatten()
            jointStateMsg.position = newPose
            jointCommandPub.publish(jointStateMsg)
        elif key == ord("s"):  # show map info
            if runId < runs:
                tmp_error = []
                diff = digitImage0.astype("float32") - referenceImage
                tactile_state = np.array(
                    pose_estimator.get_pose(diff, digitPoseImage0)
                ).reshape(1, -1)
                tmp_error.append(
                    np.array(
                        [
                            0.0,
                            np.sqrt(
                                np.square((tactile_state[0, 1] - 0.5) * image_shape[1])
                                + np.square(
                                    (tactile_state[0, 2] - 0.5) * image_shape[0]
                                )
                            ),
                        ]
                    ).reshape((1, -1))
                )
                for _ in range(actionPerRun):
                    cv2.imshow("Pose image", digitPoseImage0)
                    key = cv2.waitKey(1)
                    hand_state = np.hstack(
                        [
                            np.array(currentJointPose)[:4],
                            np.array(currentJointPose)[-4:],
                        ]
                    ).reshape(1, -1)
                    diff = digitImage0.astype("float32") - referenceImage
                    tactile_state = np.array(
                        pose_estimator.get_pose(diff, digitPoseImage0)
                    ).reshape(1, -1)
                    if withGradientOpt:
                        new_actions, outputs = ensemble_dynamics_model.evolveActions(
                            hand_state,
                            tactile_state,
                            actions,
                            np.array((0.5, 0.5, 0.0)),
                        )
                    else:
                        outputs = ensemble_dynamics_model(
                            hand_state, tactile_state, actions
                        )
                        new_actions = actions.copy()
                    contactStrength = np.mean(outputs[:, :, 0], axis=0) - 1.64 * np.std(
                        outputs[:, :, 0], axis=0
                    )  # greater than 5% chance of falling
                    filtered_outputs = np.delete(
                        outputs, np.where(contactStrength < 1.8), axis=1
                    )
                    filtered_actions = np.delete(
                        new_actions, np.where(contactStrength < 1.8), axis=0
                    )
                    if filtered_actions.shape[0] == 0:
                        print("No action is safe!! please adjust manully")
                        breakFlag = True
                        break
                    observations = (
                        filtered_outputs[:, :, 1:3]
                        * np.array([image_shape[1], image_shape[0]]).reshape((1, 1, 2))
                    ).transpose(1, 2, 0)
                    N = observations.shape[2]
                    m1 = observations - observations.sum(2, keepdims=1) / N
                    covariances = np.einsum("ijk,ilk->ijl", m1, m1) / (N - 1)
                    peaks = 1.0 / np.sqrt(
                        covariances[:, 0, 0] * covariances[:, 1, 1]
                        - covariances[:, 0, 1] * covariances[:, 1, 0]
                    )
                    filtered_outputs = np.delete(
                        filtered_outputs, np.where(peaks < peaks.max() * 0.7), axis=1
                    )
                    filtered_actions = np.delete(
                        filtered_actions, np.where(peaks < peaks.max() * 0.7), axis=0
                    )
                    location_mean = np.mean(filtered_outputs[:, :, 1:3], axis=0)
                    current_loss = np.sqrt(
                        np.square((tactile_state[0, 1] - 0.5) * image_shape[1])
                        + np.square((tactile_state[0, 2] - 0.5) * image_shape[0])
                    )
                    location_loss = np.sqrt(
                        np.square((location_mean[:, 0] - 0.5) * image_shape[1])
                        + np.square((location_mean[:, 1] - 0.5) * image_shape[0])
                    )
                    predicted_loss = np.min(location_loss)
                    if predicted_loss > current_loss:
                        print("No better action than current pose")
                        predicted_loss = current_loss
                    else:
                        centerAction = filtered_actions[np.argmin(location_loss), :]
                        newPose[:4] = centerAction[:4].flatten()
                        newPose[-3:] = centerAction[-3:].flatten()
                        jointStateMsg.header.stamp = rospy.Time.now()
                        jointStateMsg.position = newPose
                        jointCommandPub.publish(jointStateMsg)
                    rospy.sleep(1.5)
                    diff = digitImage0.astype("float32") - referenceImage
                    tactile_state = np.array(
                        pose_estimator.get_pose(diff, digitPoseImage0)
                    ).reshape(1, -1)
                    actual_loss = np.sqrt(
                        np.square((tactile_state[0, 1] - 0.5) * image_shape[1])
                        + np.square((tactile_state[0, 2] - 0.5) * image_shape[0])
                    )
                    tmp_error.append(
                        np.array([predicted_loss, actual_loss]).reshape((1, -1))
                    )
                    print("Loss is ", predicted_loss, actual_loss)
                if not breakFlag:
                    errors.append(np.vstack(tmp_error))
                    if withGradientOpt:
                        np.save(
                            "data/poseRegErrorsOpt" + str(runId) + ".npy",
                            np.vstack(tmp_error),
                        )
                    else:
                        np.save(
                            "data/poseRegErrorsSamp" + str(runId) + ".npy",
                            np.vstack(tmp_error),
                        )
                    runId += 1
                    print(
                        "Please place the object and do next..............................",
                        runId,
                        "/",
                        runs,
                    )
                    jointStateMsg.header.stamp = rospy.Time.now()
                    sample = actions[np.random.randint(0, actions.shape[0]), :]
                    newPose[:4] = sample[:4].flatten()
                    newPose[-3:] = sample[-3:].flatten()
                    jointStateMsg.position = newPose
                    jointCommandPub.publish(jointStateMsg)
                breakFlag = False
            else:
                print("Finished all runs..............................")
                np.save("data/poseRegErrors.npy", np.vstack(errors))
                break
        elif key == ord("c"):
            start_time = time.time()
            while time.time() - start_time < 5.0:
                cv2.imshow("Pose image", digitPoseImage0)
                key = cv2.waitKey(1)
                diff = digitImage0.astype("float32") - referenceImage
                tactile_state = np.array(
                    pose_estimator.get_pose(diff, digitPoseImage0)
                ).reshape(1, -1)
                print(
                    "Current error is ",
                    np.sqrt(
                        np.square((tactile_state[0, 1] - 0.5) * image_shape[1])
                        + np.square((tactile_state[0, 2] - 0.5) * image_shape[0])
                    ),
                )
