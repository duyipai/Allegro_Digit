#!/usr/bin/env python3
import os

import cv2
import numpy as np
import rospy
import torch

from allegro_hand.scripts.actionSampler import sampleActionSingle, sampleActionParallel
from cv_bridge import CvBridge

from Kinematics import FKSolver
from processData import DynamicsModel
from rospy.numpy_msg import numpy_msg
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, JointState

from digit.msg import Floats

JointStateTopic = "/allegroHand_0/joint_states"
JointCommandTopic = "/allegroHand_0/joint_cmd"
DigitTopic0 = "digit_sensor/0/raw"
DigitTopic1 = "digit_sensor/1/raw"
DigitDepthTopic0 = "digit_sensor/0/depth"
DigitDepthTopic1 = "digit_sensor/1/depth"
digitImage0 = np.zeros((320, 240), dtype=np.uint8)
digitImage1 = np.zeros((320, 240), dtype=np.uint8)
digitDepthImage0 = np.zeros((320, 240), dtype=np.float32)
digitDepthImage1 = np.zeros((320, 240), dtype=np.float32)
currentJointPose = None
currentJointTorque = None
InitialGraspPose = np.array(
    [
        0.0,
        -0.17453292519,
        0.78539816339,
        0.78539816339,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.2,
        0.1511395985137535,
        0.5,
        0.0900086861635857,
    ]
)
InitialGraspPose = [
    0.22,
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
    1.084,
    0.151,
    0.58,
    -0.1,
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
thumbFK = FKSolver("dummy", "link_15_tip")
indexFK = FKSolver("dummy", "link_3_tip")


def jointStateCallback(msg):
    global currentJointPose, currentJointTorque, thumbFK, indexFK
    currentJointPose = msg.position
    currentJointTorque = msg.effort


def DigitCallback0(msg):
    global digitImage0
    digitImage0 = bridge.imgmsg_to_cv2(msg, "passthrough")


def DigitCallback1(msg):
    global digitImage1
    digitImage1 = bridge.imgmsg_to_cv2(msg, "passthrough")


def DigitDepthCallback0(msg):
    global digitDepthImage0
    digitDepthImage0 = msg.data.reshape((320, 240))


def DigitDepthCallback1(msg):
    global digitDepthImage1
    digitDepthImage1 = msg.data.reshape((320, 240))


def evolveActions(
    model, actions, pose0, pose1, poseDesired0, prevJointPose, prevJointTorque
):
    actions = torch.from_numpy(actions).float().cuda()
    actions.requires_grad = True
    states = (
        torch.from_numpy(np.concatenate((prevJointPose, prevJointTorque, pose0, pose1)))
        .float()
        .cuda()
        .reshape(1, 22)
    )
    states = torch.tile(states, (actions.shape[0], 1))
    poseDesired0 = torch.from_numpy(poseDesired0).float().cuda().reshape(1, 3)
    poseDesired0 = torch.tile(poseDesired0, (actions.shape[0], 1))
    optimizer = torch.optim.Adam([actions], lr=0.01)
    model = model.eval()
    optimize_steps = 100
    for _ in range(optimize_steps):
        optimizer.zero_grad()
        outputs = model(states, actions)
        loss = torch.pow(outputs - poseDesired0, 2).mean(dim=1)
        print("Min loss is: ", loss.min().item(), "Max loss is: ", loss.max().item())
        loss.mean().backward()
        optimizer.step()
    with torch.no_grad():
        outputs = model(states, actions)
        loss = torch.pow(outputs - poseDesired0, 2).mean(dim=1)
        action = actions[torch.where(loss == loss.min()), :].detach().cpu().numpy()
    return action, loss.min().item()


if __name__ == "__main__":
    depthRange = 0.004
    depthContactThs = 0.001

    model_path = os.path.join("data", "dynamics_model.pth")
    model = torch.load(model_path)
    actions = sampleActionParallel(128, np.array([1.16565, 0.1846, 0.269, -0.007]))
    action, loss_pred = evolveActions(
        model,
        np.hstack((actions[:,:4], actions[:,-2:])),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(8),
        np.zeros(8),
    )

    jointStateSub = rospy.Subscriber(JointStateTopic, JointState, jointStateCallback)
    DigitSub0 = rospy.Subscriber(DigitTopic0, Image, DigitCallback0)
    DigitSub1 = rospy.Subscriber(DigitTopic1, Image, DigitCallback1)
    DigitDepthsub0 = rospy.Subscriber(
        DigitDepthTopic0, numpy_msg(Floats), DigitDepthCallback0
    )
    DigitDepthsub1 = rospy.Subscriber(
        DigitDepthTopic1, numpy_msg(Floats), DigitDepthCallback1
    )

    jointCommandPub = rospy.Publisher(JointCommandTopic, JointState, queue_size=1)
    rospy.sleep(1.0)

    jointStateMsg = JointState()

    while True:
        cv2.imshow("Digit0", digitImage0)
        cv2.imshow("DigitDepth 0", digitDepthImage0 / depthRange)
        key = cv2.waitKey()
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
        elif key == ord("j"):
            # print(
            #     "Current joint pose: ",
            #     currentJointPose[-4:],
            #     "Should be: ",
            #     InitialGraspPose[-4:],
            # )
            jointStateMsg.header.stamp = rospy.Time.now()
            # sample = sampleAction(1, [1.136, 0.17, 0.46, 0.024])
            samples = sampleActionParallel(128, currentJointPose[-4:])
            print("sample is ", sample)
            newPose[:4] = sample[0, :4].flatten()
            newPose[-4:-1] = sample[0, -4:-1].flatten()
            jointStateMsg.position = newPose
            jointCommandPub.publish(jointStateMsg)
