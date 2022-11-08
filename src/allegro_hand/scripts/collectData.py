#!/usr/bin/env python3
import os
from datetime import datetime

import cv2
import numpy as np
import rospy
from actionSampler import obtainActionSamples
from sensor_msgs.msg import Image, JointState

JointStateTopic = "/allegroHand_0/joint_states"
JointCommandTopic = "/allegroHand_0/joint_cmd"
DigitTopic0 = "digit_sensor/0/raw"
DigitTopic1 = "digit_sensor/1/raw"
digitImage0 = np.zeros((480, 640), dtype=np.uint8)
digitImage1 = np.zeros((480, 640), dtype=np.uint8)
currentJointPose = None
currentJointTorque = None
InitialGraspPose = [
    0.1798756136945422,
    1.6,
    0.16865496003106606,
    0.05466586327025375,
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


def jointStateCallback(msg):
    global currentJointPose, currentJointTorque
    currentJointPose = msg.position
    currentJointTorque = msg.effort


def DigitCallback0(msg):
    global digitImage0
    digitImage0 = msg.data


def DigitCallback1(msg):
    global digitImage1
    digitImage1 = msg.data


if __name__ == "__main__":
    sampledActionLength = 100
    actionStepSize = 0.1
    sampledActions = np.hstack((obtainActionSamples(sampledActionLength, actionStepSize),obtainActionSamples(sampledActionLength, actionStepSize)))
    data_folder = os.path.join("data", datetime.now().strftime("%b-%d-%H:%M"))
    os.makedirs(data_folder)

    rospy.init_node("collect_data")
    rospy.loginfo("Collecting data...")
    rospy.loginfo("Press Ctrl+C to stop.")

    jointStateSub = rospy.Subscriber(JointCommandTopic, JointState, jointStateCallback)
    DigitSub0 = rospy.Subscriber(DigitTopic0, Image, DigitCallback0)
    DigitSub1 = rospy.Subscriber(DigitTopic1, Image, DigitCallback1)

    jointCommandPub = rospy.Publisher(JointCommandTopic, JointState, queue_size=1)
    rospy.sleep(1.0)

    jointStateMsg = JointState()

    while True:
        cv2.imshow("Digit 0", digitImage0)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("g"):
            jointStateMsg.header.stamp = rospy.Time.now()
            jointStateMsg.position = InitialGraspPose
            jointCommandPub.publish(jointStateMsg)
        elif key == ord("h"):
            jointStateMsg.header.stamp = rospy.Time.now()
            jointStateMsg.position = HomePose
            jointCommandPub.publish(jointStateMsg)
