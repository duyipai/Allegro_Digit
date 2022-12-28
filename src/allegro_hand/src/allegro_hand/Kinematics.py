#!/usr/bin/env python

from trac_ik_python.trac_ik import IK
from urdf_parser_py.urdf import URDF
import kinpy as kp
import rospy
import numpy as np


class IKSolver:
    def __init__(self, base_link, tip_link, timeout=0.01):
        self.ik_solver = IK(
            base_link, tip_link, timeout=timeout  # , solve_type="Manipulation2"
        )
        print("IK Solver initialized")
        print("Joint limits for ", tip_link, self.ik_solver.get_joint_limits())

    def solve(self, pose, tolerance, qinit):
        if len(pose) == 3:  # position only
            return self.ik_solver.get_ik(
                qinit,
                pose[0],
                pose[1],
                pose[2],
                0.0,
                0.0,
                0.0,
                1.0,
                tolerance[0],
                tolerance[1],
                tolerance[2],
                1000,
                1000,
                1000,
            )
        elif len(pose) == 7:
            return self.ik_solver.get_ik(
                qinit,
                pose[0],
                pose[1],
                pose[2],
                pose[4],
                pose[5],
                pose[6],
                pose[3],  # note the order of the quaternion is non-ordinary in kinpy!
                tolerance[0],
                tolerance[1],
                tolerance[2],
                tolerance[3],
                tolerance[4],
                tolerance[5],
            )
        else:
            raise ValueError(
                "Pose should be of length 3 or 7 (with orientationTolerance)"
            )


class FKSolver:
    def __init__(self, base_link, tip_link):
        self.robot = URDF.from_parameter_server()
        self.base_link = base_link
        self.tip_link = tip_link
        self.joints = self.robot.get_chain(
            self.base_link, self.tip_link, links=False, joints=True, fixed=False
        )
        self.chain = kp.build_chain_from_urdf(rospy.get_param("/robot_description"))
        print(self.chain)
        self.states = {}

    def solve(self, q):
        if len(q) != len(self.joints):
            raise ValueError("Invalid number of joints")
        for i in range(len(self.joints)):
            self.states[self.joints[i]] = q[i]
        transforms = self.chain.forward_kinematics(self.states)
        result = transforms[self.base_link].inverse() * transforms[self.tip_link]
        return result


if __name__ == "__main__":
    # print(URDF.from_parameter_server())
    fk = FKSolver("dummy", "link_3_tip")
    ik = IKSolver("dummy", "link_3_tip")
    q = [0.2] * 4
    print(fk.solve(q))
    print(
        ik.solve(np.append(fk.solve(q).pos, fk.solve(q).rot), [0.01] * 6, [0.0] * 4),
        fk.solve(
            ik.solve(np.append(fk.solve(q).pos, fk.solve(q).rot), [0.01] * 6, [0.0] * 4)
        ),
    )
