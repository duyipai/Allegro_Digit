#!/usr/bin/env python3
import numpy as np
import pytact
import rospy
import torch
from cv_bridge import CvBridge
from digit.msg import Floats
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import CompressedImage, Image

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

    depth_range = 0.002
    depth_bias = 0.0004
    if sensor.is_running():
        for i in range(100):  # skip first 100 frames to give sensor time to warm up
            _ = sensor.get_frame()
        sensor.set_reference(sensor.get_frame())
    print("Finished setting reference frame")

    bridge = CvBridge()
    while not rospy.is_shutdown() and sensor.is_running():
        frame = sensor.get_frame()
        if frame is not None:
            diff = sensor.preprocess_for(lookupTable.model.model_type, frame)
            depth = lookupTable(diff)
            raw_msg = (
                bridge.cv2_to_compressed_imgmsg(frame.image, dst_format="jpg")
                if use_compressed
                else bridge.cv2_to_imgmsg(frame.image, encoding="passthrough")
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
                import cv2

                cv2.imshow("raw", frame.image)
                cv2.imshow("diff", diff)
                cv2.imshow(
                    "depth",
                    ((depth_bias - depth.data) / depth_range * 255.0).astype(np.uint8),
                )
                cv2.waitKey(1)
