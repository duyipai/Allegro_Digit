import time
from multiprocessing import Pool

import numpy as np
from kinpy import Transform
from scipy.spatial.transform import Rotation as R

from allegro_hand.Kinematics import FKSolver, IKSolver

thumbFK = FKSolver("dummy", "link_15_tip")
# thumbIK = IKSolver("dummy", "link_15_tip")
indexIK = IKSolver("dummy", "link_3_tip")
indexFK = FKSolver("dummy", "link_3_tip")


def sampleActionSingle(numSamples, thumbJointPose, trials=200):
    global thumbFK, indexIK

    # indexMotionWidth = 0.03
    indexLower = [0.01, -0.01, -0.01]
    indexUpper = [0.035, 0.02, 0.03]
    rotIndexToThumb = Transform([0.2, 0.225, 3.09])
    indexTolerance = (0.001, 0.001, 0.001, 0.3, 0.3, 0.5)
    # indexTolerance = (0.001, 0.001, 0.001, 0.15, 0.125, 0.3)
    thumbJointMin = (0.9, -0.2, -0.2, -0.3)
    thumbJointMax = (1.2, 0.4, 1.0, 0.8)
    qIndexInit = np.zeros(4)  # TODO: Find a better initial guess
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
            sampledThumbJointPose[0] = thumbJointPose[
                0
            ]  # do not change the thumb joint 0
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


def isValidAction(trans):
    indexLower = [0.01, -0.01, -0.01]
    indexUpper = [0.035, 0.02, 0.03]
    rotIndexToThumb = [0.2, 0.225, 3.09]
    indexTolerance = (0.3, 0.3, 0.5)
    rot = R.from_quat(
        [
            trans.rot[1],
            trans.rot[2],
            trans.rot[3],
            trans.rot[0],
        ]
    ).as_euler("xyz", degrees=False)
    if (
        rot[0] < rotIndexToThumb[0] - indexTolerance[0]
        or rot[0] > rotIndexToThumb[0] + indexTolerance[0]
    ):
        return False
    if (
        rot[1] < rotIndexToThumb[1] - indexTolerance[1]
        or rot[1] > rotIndexToThumb[1] + indexTolerance[1]
    ):
        return False
    if rot[2] > 0.0 and rot[2] < rotIndexToThumb[2] - indexTolerance[2]:
        return False
    if rot[2] < 0.0 and rot[2] > rotIndexToThumb[2] + indexTolerance[2] - np.pi * 2:
        return False
    pos = trans.pos
    if pos[0] < indexLower[0] or pos[0] > indexUpper[0]:
        return False
    if pos[1] < indexLower[1] or pos[1] > indexUpper[1]:
        return False
    if pos[2] < indexLower[2] or pos[2] > indexUpper[2]:
        return False
    return True


def singleProcess(
    thumbFirstJointTrue,
    thumbSecond,
    thumbJointMin,
    thumbJointMax,
    indexJointMin,
    indexJointMax,
    thumbFK,
    indexFK,
):
    for thumbThird in np.arange(thumbJointMin[2], thumbJointMax[2] + 0.05, 0.05):
        for thumbFourth in np.arange(thumbJointMin[3], thumbJointMax[3] + 0.05, 0.05):
            thumbJointPose = np.array(
                [thumbFirstJointTrue, thumbSecond, thumbThird, thumbFourth]
            )
            for indexFirst in np.arange(indexJointMin[0], indexJointMax[0], 0.05):
                for indexSecond in np.arange(indexJointMin[1], indexJointMax[1], 0.05):
                    for indexThird in np.arange(
                        indexJointMin[2], indexJointMax[2], 0.05
                    ):
                        for indexFourth in np.arange(
                            indexJointMin[3], indexJointMax[3], 0.05
                        ):
                            indexJointPose = np.array(
                                [indexFirst, indexSecond, indexThird, indexFourth]
                            )
                            trans = thumbFK.solve(
                                thumbJointPose
                            ).inverse() * indexFK.solve(
                                indexJointPose  # transformation from index frame to thumb frame
                            )
                            if isValidAction(trans):
                                action_list.append(
                                    np.hstack(
                                        (
                                            indexJointPose.reshape((1, -1)),
                                            thumbJointPose.reshape((1, -1)),
                                        )
                                    )
                                )
            print("Finished one round for thumb ", thumbJointPose)
    if len(action_list) == 0:
        return None
    else:
        return np.vstack(action_list)


if __name__ == "__main__":
    # result = sampleActionParallel(128, np.array([1.16565, 0.1846, 0.269, -0.007]))
    # print(result.shape)
    thumbJointMin = (0.9, -0.2, -0.0, -0.3)
    thumbJointMax = (1.2, 0.4, 0.8, 0.6)
    indexJointMin = (0.05, 0.95, -0.27, -0.27)
    indexJointMax = (0.35, 1.65, 0.4, 0.8)
    # fake
    indexJointMin = (-0.1, -0.1, -0.1, -0.22)
    indexJointMax = (0.1, 0.0, 0.0, 0.0)
    thumbJointMin = (0.9, -0.2, -0.2, -0.3)
    thumbJointMax = (1.2, 0.2, 0.2, 0.3)
    # end fake
    thumbFirstJointCommand = 1.05
    thumbFirstJointTrue = 1.11
    numProcesses = 6

    thumbSeconds = np.arange(thumbJointMin[1], thumbJointMax[1] + 0.05, 0.05)
    chuncks = []
    while thumbSeconds.shape[0] > numProcesses:
        chuncks.append(thumbSeconds[:numProcesses])
        thumbSeconds = thumbSeconds[numProcesses:]
    chuncks.append(thumbSeconds)

    action_list = []
    i = 0
    t_start = time.time()
    for chunck in chuncks:
        p = Pool(numProcesses)
        results_objects = []
        for i in range(chunck.shape[0]):
            thumbSecond = chunck[i]
            results_objects.append(
                p.apply_async(
                    singleProcess,
                    args=(
                        thumbFirstJointTrue,
                        thumbSecond,
                        thumbJointMin,
                        thumbJointMax,
                        indexJointMin,
                        indexJointMax,
                        thumbFK,
                        indexFK,
                    ),
                )
            )
        p.close()
        p.join()
        samples = []
        for obj in results_objects:
            if obj.get() is not None:
                samples.append(obj.get())
        if len(samples) > 0:
            action_list.append(np.vstack(samples))
        print("Finished one round for thumb second joint", chunck)
    print("Time used ", time.time() - t_start)
    if len(action_list) > 0:
        action_list = np.vstack(action_list)
        np.save("actions", action_list)
