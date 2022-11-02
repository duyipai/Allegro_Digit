#!/usr/bin/env python3
from math import acos, pi

import cv2
import numba
import numpy as np
import pytact
import rospy
import torch
from cv_bridge import CvBridge
from digit.msg import Floats
from numpy import linalg as LA
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import CompressedImage, Image


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


class frameMedianFilter:
    def __init__(self, sample_frame, size=5):
        self.frames = []
        for _ in range(size):
            self.frames.append(sample_frame.image)

    def __call__(self, frame):
        self.frames.append(frame.image)
        self.frames.pop(0)
        filtered = frameMedianFilter.filter(self.frames)
        return pytact.types.Frame(pytact.types.FrameEnc.BGR, filtered.astype(np.uint8))

    def filter(frames):
        return np.median(
            np.stack(frames, axis=3),
            axis=3,
            keepdims=False,
        )


RAW_TOPIC = "digit_sensor/raw"
DIFF_TOPIC = "digit_sensor/diff"
DEPTH_TOPIC = "digit_sensor/depth"

if __name__ == "__main__":
    rospy.init_node("digit_sensor_pub")

    scale = rospy.get_param("~scale")
    calculate_depth = rospy.get_param("~calculate_depth", False)

    # Depth related params
    model_path = rospy.get_param("~model_path")
    mmpp = rospy.get_param("~mmpp")
    device = "cuda" if rospy.get_param("~use_gpu") else "cpu"
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
    do_pca = rospy.get_param("~do_pca", False)

    serial = rospy.get_param("~serial")
    use_compressed = rospy.get_param("~use_compressed", True)
    visualize = rospy.get_param("~visualize", False)
    sensor = pytact.sensors.sensor_from_args(
        **{"sensor_name": "DigitV2", "serial": serial}
    )

    if use_compressed:
        RAW_TOPIC += "/compressed"
        DIFF_TOPIC += "/compressed"
        raw_pub = rospy.Publisher(RAW_TOPIC, CompressedImage, queue_size=1)
        diff_pub = rospy.Publisher(DIFF_TOPIC, CompressedImage, queue_size=1)
    else:
        raw_pub = rospy.Publisher(RAW_TOPIC, Image, queue_size=1)
        diff_pub = rospy.Publisher(DIFF_TOPIC, Image, queue_size=1)
    if calculate_depth:
        depth_pub = rospy.Publisher(DEPTH_TOPIC, numpy_msg(Floats), queue_size=1)
        depth_range = 0.0004
        depth_bias = 0.0002

    median_filter = frameMedianFilter(sensor.get_frame())
    if sensor.is_running():
        for i in range(100):  # skip first 100 frames to give sensor time to warm up
            f = median_filter(sensor.get_frame())
        sensor.set_reference(f)
    print("Finished setting reference frame")

    bridge = CvBridge()
    pose_estimator = Pose()
    while not rospy.is_shutdown() and sensor.is_running():
        frame = median_filter(sensor.get_frame())
        if frame is not None:
            f = frame.image
            diff = sensor.preprocess_for(lookupTable.model.model_type, frame)
            diff_msg = np.abs(diff.image).astype(np.uint8)
            diff_msg = (
                bridge.cv2_to_compressed_imgmsg(diff_msg, dst_format="jpg")
                if use_compressed
                else bridge.cv2_to_imgmsg(diff_msg, encoding="passthrough")
            )
            diff_pub.publish(diff_msg)

            if calculate_depth:
                depth = lookupTable(diff)
                depth.data[np.where(depth.data > 0.0)] = 0.0
                depth.data = -depth.data

                if do_pca:
                    pose_estimator.get_pose(depth.data, frame.image)
                    f = pose_estimator.frame

                depth_pub.publish(depth.data)

            raw_msg = (
                bridge.cv2_to_compressed_imgmsg(f, dst_format="jpg")
                if use_compressed
                else bridge.cv2_to_imgmsg(f, encoding="passthrough")
            )
            raw_pub.publish(raw_msg)

            if visualize:
                cv2.imshow("raw", f)
                cv2.imshow("diff", np.abs(diff.image).astype(np.uint8) * 3)
                # print(
                #     "depth min: {}, max: {}".format(depth.data.min(), depth.data.max())
                # )
                if calculate_depth:
                    cv2.imshow(
                        "depth",
                        ((depth.data) / depth_range * 255.0).astype(np.uint8),
                    )
                cv2.waitKey(1)
