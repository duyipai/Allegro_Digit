from math import acos, pi

import cv2
import numpy as np
from numpy import linalg as LA


class Pose:
    def __init__(self):
        self.pose = None

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
            theta / pi * 180,
            0,
            360,
            (255, 255, 255),
            lineThickness,
        )

    def get_pose(self, diff, frame):
        self.diff = np.abs(diff).sum(axis=2)
        thresh = 30
        # print("diff mean max", self.diff.mean(), self.diff.max())

        mask = self.diff > thresh
        cv2.imshow("mask", mask.astype(np.uint8) * 255)
        # Display the resulting frame
        coors = np.where(mask == 1)
        X = coors[1].reshape(-1, 1)
        y = coors[0].reshape(-1, 1)

        if len(X) > 10:
            pts = np.concatenate([X, y], axis=1)
            v_max, v_min, w_max, w_min = self.PCA(pts)
            if v_max[1] < 0:
                v_max *= -1
            m = np.mean(pts, 0).reshape(-1)

            # Record pose estimation
            self.pose = (v_max, v_min, w_max, w_min, m)

            self.draw_ellipse(frame, self.pose)
            rho = (
                (v_max[0] * m[1] - v_max[1] * m[0])
                / (v_max[0] ** 2 + v_max[1] ** 2) ** 0.5
            ).item()
            theta = acos(v_max[0] / (v_max[0] ** 2 + v_max[1] ** 2) ** 0.5)
            return (
                mask.mean(),
                rho / np.hypot(frame.shape[0], frame.shape[1]),
                theta / np.pi,
            )
        else:
            # No contact
            self.pose = None
            self.frame = frame
            return (
                0.0,
                0.0,
                0.5,
            )
