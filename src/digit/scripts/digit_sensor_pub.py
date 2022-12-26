#!/usr/bin/env python3

import cv2
import numpy as np
import pytact
import rospy
import torch
from cv_bridge import CvBridge
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import CompressedImage, Image

from digit.msg import Floats
from digit.pose_estimation import Pose


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
        depth_range = 0.003

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

                depth_pub.publish(depth.data.astype(np.float32).flatten())

            if do_pca:
                print(pose_estimator.get_pose(diff.image, frame.image))
                f = pose_estimator.frame

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
