from Kinematics import IKSolver, FKSolver
import numpy as np
from kinpy import Transform
from multiprocessing import Pool

thumbFK = FKSolver("dummy", "link_15_tip")
thumbIK = IKSolver("dummy", "link_15_tip")
indexIK = IKSolver("dummy", "link_3_tip")
indexfk = FKSolver("dummy", "link_3_tip")

# indexMotionWidth = 0.03
indexLower = [0.01, -0.02, -0.005]
indexUpper = [0.04, 0.02, 0.03]
rotIndexToThumb = Transform([0.225, 0.45, 3.14])
# indexTolerance = (0.001, 0.001, 0.001, 0.225, 0.08, 0.4)
indexTolerance = (0.001, 0.001, 0.001, 0.3, 0.2, 0.4)
thumbJointMin = (0.92, -0.38, 0.0, -0.0)
thumbJointMax = (1.12, 0.616, 1.0, 0.7)
qIndexInit = np.zeros(4)  # TODO: Find a better initial guess


def sampleActionSingle(numSamples, thumbJointPose, trials=2000):
    global thumbFK, thumbIK, indexIK, indexLower, indexUpper, thumbJointMax, thumbJointMin
    # thumbPos = thumbFK.solve(thumbJointPose)
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
            sampledThumbPose = thumbFK.solve(sampledThumbJointPose)
            # print(
            #     "Thumb joint pose and new pose", thumbJointPose, sampledThumbJointPose
            # )
            # print("Thumb pos and new pos", thumbPos, sampledThumbPose)

            u = np.random.uniform(indexLower[0], indexUpper[0])
            v = np.random.uniform(indexLower[1], indexUpper[1])
            w = np.random.uniform(indexLower[2], indexUpper[2])
            # print("u, v, w", u, v, w)
            sampledIndexPose = np.append(
                (
                    sampledThumbPose.matrix() @ np.array([u, v, w, 1]).reshape(4, 1)
                ).flatten()[:3],
                (sampledThumbPose * rotIndexToThumb).rot,
            )
            # print("Sampled index pos", sampledIndexPose)
            sampledIndexJointPose = indexIK.solve(
                sampledIndexPose, indexTolerance, qIndexInit
            )
            if sampledIndexJointPose is not None:
                # obtainedIndexPos = indexfk.solve(sampledIndexJointPose)
                # print("Obtained index pos", obtainedIndexPos)
                samples.append(
                    np.hstack((sampledIndexJointPose, sampledThumbJointPose))
                )
                break
            i += 1
            if i == trials:
                samples.append(np.zeros(8))
    return np.vstack(samples)


def sampleActionParallel(numSamples, thumbJointPose, numProcesses=4):
    p = Pool(numProcesses)
    subSampleSize = int(numSamples / numProcesses + 0.5)
    results_objects = []
    for i in range(numProcesses):
        results_objects.append(
            p.apply_async(
                sampleActionSingle,
                args=(subSampleSize, thumbJointPose),
            )
        )
    p.close()
    p.join()
    samples = []
    for obj in results_objects:
        samples.append(obj.get())
    samples = np.vstack(samples)
    samples = samples[~np.all(samples == 0.0, axis=1)]
    return samples


if __name__ == "__main__":
    result = sampleActionParallel(128, np.array([1.16565, 0.1846, 0.269, -0.007]))
    print(result.shape)
