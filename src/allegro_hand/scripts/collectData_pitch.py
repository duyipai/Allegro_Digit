#!/usr/bin/env python3
import os
from datetime import datetime

import cv2
import numpy as np
import rospy
from actionSampler_pitch import sampleAction
from cv_bridge import CvBridge
from Kinematics import FKSolver
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image, JointState

from digit.msg import Floats
from scipy.spatial.transform import Rotation as R

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
    # trans = thumbFK.solve([1.136, 0.17, 0.46, 0.024]).inverse() * indexFK.solve(
    #     currentJointPose[:4]
    # )
    # trans = thumbFK.solve(currentJointPose[-4:]).inverse() * indexFK.solve(
    #     currentJointPose[:4]
    # )
    # print(
    #     R.from_quat([trans.rot[1], trans.rot[2], trans.rot[3], trans.rot[0]]).as_euler(
    #         "xyz", degrees=False
    #     )
    # )
    print(currentJointPose[:4], currentJointPose[-4:])
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


if __name__ == "__main__":
    depthRange = 0.004
    depthContactThs = 0.001
    samplingAmount = 10000

    if not os.path.exists("data"):
        os.makedirs("data")
    data_folder = os.path.join("data", datetime.now().strftime("%d-%H:%M:%S"))
    os.makedirs(data_folder)

    rospy.init_node("collect_data")
    rospy.loginfo("Collecting data...")
    rospy.loginfo("Press Ctrl+C to stop.")

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

    for i in range(samplingAmount):
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
            i -= 1
        elif key == ord("h"):
            jointStateMsg.header.stamp = rospy.Time.now()
            jointStateMsg.position = HomePose
            jointCommandPub.publish(jointStateMsg)
            i -= 1
        elif key == ord("j"):
            # print(
            #     "Current joint pose: ",
            #     currentJointPose[-4:],
            #     "Should be: ",
            #     InitialGraspPose[-4:],
            # )
            jointStateMsg.header.stamp = rospy.Time.now()
            # sample = sampleAction(1, [1.136, 0.17, 0.46, 0.024])
            sample = sampleAction(1, currentJointPose[-4:])
            print("sample is ", sample)
            newPose[:4] = sample[0, :4].flatten()
            newPose[-4:-1] = sample[0, -4:-1].flatten()
            jointStateMsg.position = newPose
            jointCommandPub.publish(jointStateMsg)
