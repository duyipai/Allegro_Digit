#!/usr/bin/env python3
import numpy as np
import pytact
import rospy
import torch
from cv_bridge import CvBridge
from digit.msg import Floats
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import CompressedImage, Image

from math import pi, acos
import cv2
from numpy import linalg as LA


class Pose:
    def __init__(self):
        self.pose = None
        self.area = 0
        self.depth = None

    def PCA(self, pts):
        pts = pts.reshape(-1, 2).astype(np.float64)
        mv = np.mean(pts, 0).reshape(2, 1)
        pts -= mv.T
        w, v = LA.eig(np.dot(pts.T, pts))
        w_max = np.max(w)
        w_min = np.min(w)

        col = np.where(w == w_max)[0]
        if len(col) > 1:
            col = col[-1]
        V_max = v[:, col]

        col_min = np.where(w == w_min)[0]
        if len(col_min) > 1:
            col_min = col_min[-1]
        V_min = v[:, col_min]

        return V_max, V_min, w_max, w_min

    def draw_ellipse(self, frame, pose):
        v_max, v_min, w_max, w_min, m = pose
        lineThickness = 2
        K = 1

        w_max = w_max**0.5 / 20 * K
        w_min = w_min**0.5 / 30 * K

        v_max = v_max.reshape(-1) * w_max
        v_min = v_min.reshape(-1) * w_min

        m1 = m - v_min
        mv = m + v_min
        self.frame = cv2.line(
            frame,
            (int(m1[0]), int(m1[1])),
            (int(mv[0]), int(mv[1])),
            (0, 255, 0),
            lineThickness,
        )

        m1 = m - v_max
        mv = m + v_max

        self.frame = cv2.line(
            self.frame,
            (int(m1[0]), int(m1[1])),
            (int(mv[0]), int(mv[1])),
            (0, 0, 255),
            lineThickness,
        )

        theta = acos(v_max[0] / (v_max[0] ** 2 + v_max[1] ** 2) ** 0.5)
        length_max = w_max
        length_min = w_min
        axis = (int(length_max), int(length_min))

        self.frame = cv2.ellipse(
            self.frame,
            (int(m[0]), int(m[1])),
            axis,
            -theta / pi * 180,
            0,
            360,
            (255, 255, 255),
            lineThickness,
        )

    def get_pose(self, depth, frame):
        self.depth = depth.copy()
        thresh = max(0.00003, depth.max() / 2)

        mask = depth > thresh
        # Display the resulting frame
        coors = np.where(mask == 1)
        X = coors[1].reshape(-1, 1)
        y = coors[0].reshape(-1, 1)

        self.area = 0
        if len(X) > 10:
            pts = np.concatenate([X, y], axis=1)
            v_max, v_min, w_max, w_min = self.PCA(pts)
            if v_max[0] > 0 and v_max[1] > 0:
                v_max *= -1
            m = np.mean(pts, 0).reshape(-1)

            # Record pose estimation
            self.pose = (v_max, v_min, w_max, w_min, m)
            self.area = len(X)

            self.draw_ellipse(frame, self.pose)
        else:
            # No contact
            self.pose = None
            self.frame = frame


RAW_TOPIC = "digit_sensor/raw"
DIFF_TOPIC = "digit_sensor/diff"
DEPTH_TOPIC = "digit_sensor/depth"

if __name__ == "__main__":
    rospy.init_node("digit_sensor_pub")
    model_path = rospy.get_param("~model_path")
    serial = rospy.get_param("~serial")
    device = "cuda" if rospy.get_param("~use_gpu") else "cpu"
    scale = rospy.get_param("~scale")
    mmpp = rospy.get_param("~mmpp")
    use_compressed = rospy.get_param("~use_compressed", True)
    visualize = rospy.get_param("~visualize", False)
    do_pca = rospy.get_param("~do_pca", False)

    sensor = pytact.sensors.sensor_from_args(
        **{"sensor_name": "DigitV2", "serial": serial}
    )
    npzfile = np.load(model_path + ".npz")
    mean = npzfile["mean"]
    std = npzfile["std"]
    model = pytact.models.Pixel2GradModel().to(device)
    model.load_state_dict(torch.load(model_path + ".pth", map_location=device))
    model.eval()
    lookupTable = pytact.tasks.DepthFromLookup(
        model,
        mmpp=mmpp,
        scale=scale,
        mean=mean,
        std=std,
        use_cuda=device == "cuda",
        optimize=False,
    )

    if use_compressed:
        RAW_TOPIC += "/compressed"
        DIFF_TOPIC += "/compressed"
        raw_pub = rospy.Publisher(RAW_TOPIC, CompressedImage, queue_size=1)
        diff_pub = rospy.Publisher(DIFF_TOPIC, CompressedImage, queue_size=1)
    else:
        raw_pub = rospy.Publisher(RAW_TOPIC, Image, queue_size=1)
        diff_pub = rospy.Publisher(DIFF_TOPIC, Image, queue_size=1)
    depth_pub = rospy.Publisher(DEPTH_TOPIC, numpy_msg(Floats), queue_size=1)

    depth_range = 0.0004
    depth_bias = 0.0002
    if sensor.is_running():
        for i in range(100):  # skip first 100 frames to give sensor time to warm up
            _ = sensor.get_frame()
        sensor.set_reference(sensor.get_frame())
    print("Finished setting reference frame")

    bridge = CvBridge()
    pose_estimator = Pose()
    while not rospy.is_shutdown() and sensor.is_running():
        frame = sensor.get_frame()
        if frame is not None:
            diff = sensor.preprocess_for(lookupTable.model.model_type, frame)
            depth = lookupTable(diff)
            depth.data[np.where(depth.data > 0.0)] = 0.0
            depth.data = -depth.data

            if do_pca:
                pose_estimator.get_pose(depth.data, frame.image)
                f = pose_estimator.frame
            else:
                f = frame.image
            raw_msg = (
                bridge.cv2_to_compressed_imgmsg(f, dst_format="jpg")
                if use_compressed
                else bridge.cv2_to_imgmsg(pose_estimator.frame, encoding="passthrough")
            )
            raw_pub.publish(raw_msg)
            diff = np.abs(diff.image).astype(np.uint8)
            diff_msg = (
                bridge.cv2_to_compressed_imgmsg(diff, dst_format="jpg")
                if use_compressed
                else bridge.cv2_to_imgmsg(diff, encoding="passthrough")
            )
            diff_pub.publish(diff_msg)
            depth_pub.publish(depth.data)
            if visualize:
                cv2.imshow("raw", f)
                cv2.imshow("diff", diff * 3)
                # print(
                #     "depth min: {}, max: {}".format(depth.data.min(), depth.data.max())
                # )
                cv2.imshow(
                    "depth",
                    ((depth.data) / depth_range * 255.0).astype(np.uint8),
                )
                cv2.waitKey(1)
