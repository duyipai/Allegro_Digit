#!/usr/bin/env python3
import os
import time

import cv2
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
from matplotlib import pyplot as plt
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


def getAngleMap(means, stds):
    means = means.reshape((-1, 1))
    stds = stds.reshape((-1, 1))
    theta = np.linspace(0, np.pi, 50)
    thetas = np.tile(theta, (means.shape[0], 1))
    diff = np.stack(
        [thetas - means + np.pi, thetas - means, thetas - means - np.pi], axis=0
    )
    diff = np.abs(diff).min(axis=0)
    angleMap = np.exp(-((diff / stds) ** 2) / 2) / np.sqrt(2 * np.pi) / stds
    probabilityMap = angleMap.max(axis=0)
    actionMap = angleMap.argmax(axis=0)
    return probabilityMap, actionMap, theta


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
    ranks = [1.0]
    trial_for_each_rank = 50
    rank_index = 0
    trial_index = 0
    errors = []
    fall_time = np.zeros(len(ranks))
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
            break_flag = False
            while rank_index < len(ranks):
                while trial_index < trial_for_each_rank:
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
                    if (
                        tactile_state[0, 1] == 0.5
                        and tactile_state[0, 2] == 0.5
                        and tactile_state[0, 3] == 0.5
                    ):
                        print("Object not in hand!! Check bugs!!")
                        break_flag = True
                        break
                    outputs = ensemble_dynamics_model(
                        hand_state, tactile_state, actions
                    )
                    contactStrength = np.mean(outputs[:, :, 0], axis=0) - 1.64 * np.std(
                        outputs[:, :, 0], axis=0
                    )  # greater than 5% chance of falling
                    filtered_outputs = np.delete(
                        outputs, np.where(contactStrength < 1.8), axis=1
                    )
                    filtered_actions = np.delete(
                        actions, np.where(contactStrength < 1.8), axis=0
                    )
                    if filtered_actions.shape[0] == 0:
                        print("No action is safe!! please adjust manully")
                        break_flag = True
                        break
                    angle_mean = np.mean(filtered_outputs[:, :, 3], axis=0) * np.pi
                    angle_std = np.std(filtered_outputs[:, :, 3], axis=0) * np.pi

                    probabilityMap, actionMap, theta = getAngleMap(
                        angle_mean, angle_std
                    )
                    plt.close("all")
                    plt.figure()
                    plt.plot(theta, probabilityMap)
                    plt.show(block=False)
                    cv2.waitKey(1)
                    probability = ranks[rank_index] * np.max(probabilityMap)
                    dist = np.abs(probabilityMap - probability)
                    found_probability = dist.min()
                    loc = np.where(dist.flatten() == found_probability)
                    if found_probability > 0.05 * np.max(probabilityMap):
                        print(
                            "Action probability not found for rank ",
                            ranks[rank_index],
                            " trial ",
                            trial_index,
                            " please adjust manully",
                        )
                        break_flag = True
                        break
                    action = filtered_actions[actionMap[loc[0][0]], :]
                    target = theta[loc[0][0]]
                    jointStateMsg.header.stamp = rospy.Time.now()
                    newPose[:4] = action[:4]
                    newPose[-3:] = action[-3:]
                    jointStateMsg.position = newPose
                    jointCommandPub.publish(jointStateMsg)
                    rospy.sleep(2.0)
                    diff = digitImage0.astype("float32") - referenceImage
                    tactile_state = np.array(
                        pose_estimator.get_pose(diff, digitPoseImage0)
                    ).reshape(1, -1)
                    if (
                        tactile_state[0, 1] == 0.5
                        and tactile_state[0, 2] == 0.5
                        and tactile_state[0, 3] == 0.5
                    ):
                        print("Object falls")
                        fall_time[rank_index] += 1
                        break_flag = True
                        break
                    else:
                        errors.append(
                            np.array(
                                [
                                    min(
                                        np.abs(
                                            target + np.pi - tactile_state[0, 3] * np.pi
                                        ),
                                        np.abs(
                                            target - np.pi - tactile_state[0, 3] * np.pi
                                        ),
                                        np.abs(target - tactile_state[0, 3] * np.pi),
                                    )
                                ]
                            )
                        )
                    trial_index += 1
                if break_flag:
                    break_flag = False
                    break
                rank_index += 1
                if rank_index < len(ranks):
                    trial_index = 0
                print("Finished rank ", ranks[rank_index - 1])
            if rank_index == len(ranks) and trial_index == trial_for_each_rank:
                print("Finished all ranks")
                errors = np.vstack(errors).reshape((len(ranks), trial_for_each_rank))
                np.save(os.path.join("data", "angle_errors.npy"), errors)
                np.save(os.path.join("data", "fall_time.npy"), fall_time)
