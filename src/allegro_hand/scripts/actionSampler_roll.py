from Kinematics import IKSolver, FKSolver
import numpy as np
from kinpy import Transform

thumbFK = FKSolver("dummy", "link_15_tip")
thumbIK = IKSolver("dummy", "link_15_tip")
indexIK = IKSolver("dummy", "link_3_tip")
indexfk = FKSolver("dummy", "link_3_tip")

thumbMotionWidth = 0.2
# indexMotionWidth = 0.03
indexLower = [-0.01, -0.015, -0.005]
indexUpper = [0.03, 0.015, 0.025]
rotIndexToThumb = Transform([0.225, 0.45, 3.14])
# indexTolerance = (0.001, 0.001, 0.001, 0.225, 0.08, 0.4)
indexTolerance = (0.001, 0.001, 0.001, 0.3, 0.2, 0.4)
thumbJointMin = (0.92, -0.38, 0.0, -0.0)
thumbJointMax = (1.12, 0.616, 1.0, 0.7)


def sampleAction(numSamples, thumbJointPose, trials=2000):
    global thumbFK, thumbIK, indexIK, thumbMotionWidth, indexLower, indexUpper
    thumbPos = thumbFK.solve(thumbJointPose)
    samples = []
    for _ in range(numSamples):
        i = 0
        while i < trials:
            sampledThumbJointPose = [
                np.random.uniform(thumbJointMin[0], thumbJointMax[0]),
                np.random.uniform(thumbJointMin[1], thumbJointMax[1]),
                np.random.uniform(thumbJointMin[2], thumbJointMax[2]),
                np.random.uniform(thumbJointMin[3], thumbJointMax[3]),
            ]
            sampledThumbJointPose[:2] = thumbJointPose[:2]
            # print(
            #     "Thumb joint pose and new pose", thumbJointPose, sampledThumbJointPose
            # )
            # sampledThumbJointPose = thumbJointPose
            sampledThumbPose = thumbFK.solve(sampledThumbJointPose)
            print("Thumb pos and new pos", thumbPos, sampledThumbPose)

            u = np.random.uniform(indexLower[0], indexUpper[0])
            v = np.random.uniform(indexLower[1], indexUpper[1])
            w = np.random.uniform(indexLower[2], indexUpper[2])
            print("u, v, w", u, v, w)
            sampledIndexPose = np.append(
                (
                    sampledThumbPose.matrix() @ np.array([u, v, w, 1]).reshape(4, 1)
                ).flatten()[:3],
                (sampledThumbPose * rotIndexToThumb).rot,
            )
            print("Sampled index pos", sampledIndexPose)
            sampledIndexJointPose = indexIK.solve(
                sampledIndexPose, indexTolerance, np.zeros(4)
            )
            if sampledIndexJointPose is not None:
                obtainedIndexPos = indexfk.solve(sampledIndexJointPose)
                print("Obtained index pos", obtainedIndexPos)
                samples.append(
                    np.hstack((sampledIndexJointPose, sampledThumbJointPose))
                )
                break
            i += 1
            if i == trials:
                raise Exception("Failed to sample action")
    return np.vstack(samples)


if __name__ == "__main__":
    print(
        sampleAction(
            1,
            np.array([1.16565, 0.1846, 0.269, -0.007]),
        )
    )
