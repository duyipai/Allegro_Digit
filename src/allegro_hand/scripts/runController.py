#!/usr/bin/env python3
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
from processData import DynamicsModel
from sensor_msgs.msg import Image, JointState

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
        self, hand_state, tactile_state, actions, target, optimize_steps=100, lr=0.0001
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
                    model(new_hand_state, new_tactile_state, new_actions)[:, 1:]
                    - target
                )
                output[:, 2] = 0.5 - torch.abs(output[:, 2] - 0.5)
                outputs.append(output)
            loss = torch.abs(torch.stack(outputs, dim=0)).mean()
            print("Optimizing loss is ", loss.item())
            loss.backward()
            optimizer.step()
        new_actions = new_actions.detach().cpu().numpy()
        return new_actions, self(hand_state, tactile_state, new_actions)


def jointStateCallback(msg):
    global currentJointPose, currentJointTorque, thumbFK, indexFK
    currentJointPose = msg.position
    currentJointTorque = msg.effort


def DigitCallback0(msg):
    global digitImage0
    digitImage0 = bridge.imgmsg_to_cv2(msg, "passthrough")


def DigitPoseCallback0(msg):
    global digitPoseImage0
    digitPoseImage0 = bridge.imgmsg_to_cv2(msg, "passthrough")


def getMap(mean, cov, xy):
    # t = time.time()
    det = 2 * np.pi * np.sqrt(np.linalg.det(cov))
    inv = np.linalg.inv(cov)
    diff = xy - mean.reshape((1, 2))
    outputHotmap = np.exp(-(diff @ inv * diff).sum(axis=1) / 2) / det
    # print("Time for getMap", time.time() - t)
    return outputHotmap


def getMaps(filtered_mean, covariances, image_shape):
    xy = (
        np.array(
            np.meshgrid(
                np.linspace(0, image_shape[1] - 1, image_shape[1]),
                np.linspace(0, image_shape[0] - 1, image_shape[0]),
            )
        )
        .reshape((2, -1))
        .T
    )
    outputHotmap = []
    for i in range(filtered_mean.shape[0]):
        outputHotmap.append(
            getMap(
                filtered_mean[i, :2].reshape((1, 2)), covariances[i, :, :], xy
            ).reshape((image_shape[0], image_shape[1]))
        )
    outputHotmap = np.stack(outputHotmap, axis=0)
    actionMap = np.argmax(outputHotmap, axis=0)
    outputHotmap = np.max(outputHotmap, axis=0)
    actionMap[outputHotmap < outputHotmap.max() * 0.05] = -1
    outputHotmap[outputHotmap < outputHotmap.max() * 0.05] = 0
    return outputHotmap, actionMap


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
    upAction = None
    downAction = None
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
            hand_state = np.hstack(
                [
                    np.array(currentJointPose)[:4],
                    np.array(currentJointPose)[-4:],
                    # np.array(currentJointTorque)[:4],
                    # np.array(currentJointTorque)[-4:],
                ]
            ).reshape(1, -1)
            diff = digitImage0.astype("float32") - referenceImage
            tactile_state = np.array(
                pose_estimator.get_pose(diff, digitPoseImage0)
            ).reshape(1, -1)
            print("hand_state is ", hand_state)
            print("tactile_state is ", tactile_state)
            outputs = ensemble_dynamics_model(hand_state, tactile_state, actions)
            # new_actions, outputs = ensemble_dynamics_model.evolveActions(
            #     hand_state, tactile_state, actions, np.array((0.5, 0.5, 0.0))
            # )
            print("outputs is ", outputs.shape)
            contactStrength = np.mean(outputs[:, :, 0], axis=0) - 1.28 * np.std(
                outputs[:, :, 0], axis=0
            )  # greater than 10% chance of falling
            filtered_outputs = np.delete(
                outputs, np.where(contactStrength < 1.8), axis=1
            )
            filtered_actions = np.delete(
                actions, np.where(contactStrength < 1.8), axis=0
            )
            print("Filter with contact is ", filtered_outputs.shape)
            angle_mean = np.mean(filtered_outputs[:, :, 3], axis=0) * np.pi
            angle_std = np.std(filtered_outputs[:, :, 3], axis=0) * np.pi
            filtered_mean = np.mean(filtered_outputs[:, :, 1:], axis=0)
            filtered_mean *= np.array([image_shape[1], image_shape[0], np.pi]).reshape(
                (1, 3)
            )
            observations = (
                filtered_outputs[:, :, 1:3]
                * np.array([image_shape[1], image_shape[0]]).reshape((1, 1, 2))
            ).transpose(1, 2, 0)
            N = observations.shape[2]
            m1 = observations - observations.sum(2, keepdims=1) / N
            covariances = np.einsum("ijk,ilk->ijl", m1, m1) / (N - 1)
            print("filtered_mean is ", filtered_mean.shape)

            print("Start building hot map")
            outputHotmap, actionMap = getMaps(filtered_mean, covariances, image_shape)
            print("Finished building hot map")
            cv2.imshow(
                "outputHotmap",
                cv2.applyColorMap(
                    (outputHotmap / np.max(outputHotmap) * 255).astype(np.uint8),
                    cv2.COLORMAP_HOT,
                ),
            )
            tmp = outputHotmap.copy()
            upAction = None
            downAction = None
            centerAction = None
            min_center_loss = (
                np.abs(tactile_state[0, 1] - 0.5)
                + np.abs(tactile_state[0, 2] - 0.5)
                + np.abs(0.5 - np.abs(tactile_state[0, 3] - 0.5))
            )
            for max_count in range(40):
                max_index = np.unravel_index(np.argmax(tmp), tmp.shape)
                if actionMap[max_index] != -1:  # action exist
                    if (
                        downAction is None
                        and max_index[0] > tactile_state[0, 2] * image_shape[0]
                    ):
                        downAction = filtered_actions[actionMap[max_index], :]
                        downTarget = max_index
                    if (
                        upAction is None
                        and max_index[0] < tactile_state[0, 2] * image_shape[0]
                    ):
                        upAction = filtered_actions[actionMap[max_index], :]
                        upTarget = max_index
                    m = filtered_mean[actionMap[max_index], :]
                    center_loss = (
                        np.abs(m[0] - 0.5)
                        + np.abs(m[1] - 0.5)
                        + np.abs(0.5 - np.abs(m[2] - 0.5))
                    )
                    if center_loss < min_center_loss:
                        min_center_loss = center_loss
                        centerAction = filtered_actions[actionMap[max_index], :]
                tmp[max_index] = 0
        elif key == ord("u"):
            if upAction is not None:
                newPose[:4] = upAction[:4].flatten()
                newPose[-3:] = upAction[-3:].flatten()
                jointStateMsg.header.stamp = rospy.Time.now()
                jointStateMsg.position = newPose
                jointCommandPub.publish(jointStateMsg)
                rospy.sleep(1.0)
                diff = digitImage0.astype("float32") - referenceImage
                tactile_state = np.array(
                    pose_estimator.get_pose(diff, digitPoseImage0)
                ).reshape(1, -1)
                print(
                    "Position error is (x/y) ",
                    (
                        np.abs(tactile_state[0, 1] * image_shape[1] - upTarget[1]),
                        np.abs(tactile_state[0, 2] * image_shape[0] - upTarget[0]),
                    ),
                )
            else:
                print("No up action found")
        elif key == ord("d"):
            if downAction is not None:
                newPose[:4] = downAction[:4].flatten()
                newPose[-3:] = downAction[-3:].flatten()
                jointStateMsg.header.stamp = rospy.Time.now()
                jointStateMsg.position = newPose
                jointCommandPub.publish(jointStateMsg)
                rospy.sleep(1.0)
                diff = digitImage0.astype("float32") - referenceImage
                tactile_state = np.array(
                    pose_estimator.get_pose(diff, digitPoseImage0)
                ).reshape(1, -1)
                print(
                    "tactile error is (x/y) ",
                    (
                        np.abs(tactile_state[0, 1] * image_shape[1] - downTarget[1]),
                        np.abs(tactile_state[0, 2] * image_shape[0] - downTarget[0]),
                    ),
                )
            else:
                print("No down action found")
        elif key == ord("c"):  # go to center
            if centerAction is not None:
                newPose[:4] = centerAction[:4].flatten()
                newPose[-3:] = centerAction[-3:].flatten()
                jointStateMsg.header.stamp = rospy.Time.now()
                jointStateMsg.position = newPose
                jointCommandPub.publish(jointStateMsg)
                rospy.sleep(1.0)
                diff = digitImage0.astype("float32") - referenceImage
                tactile_state = np.array(
                    pose_estimator.get_pose(diff, digitPoseImage0)
                ).reshape(1, -1)
                print(
                    "Center loss is (x/y/rad) ",
                    (
                        np.abs(m[0] - 0.5) * image_shape[1],
                        np.abs(m[1] - 0.5) * image_shape[0],
                        np.abs(0.5 - np.abs(m[2] - 0.5)) * np.pi,
                    ),
                )
            else:
                print("No center action found")
