import numpy as np


def obtainActionSamples(numSamples, stepSize=1.0):
    """
    Obtain samples of given action with a given step size
    :param numSamples: the number of samples to obtain
    :param stepSize: the step size to use for sampling
    :return: a list of sampled actions
    """
    action = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ]
    )
    samples = np.empty((numSamples, action.shape[1]))
    inds = np.random.randint(0, action.shape[0], numSamples)
    polars = np.random.choice([-1, 1], numSamples).reshape(-1, 1)
    samples = polars * stepSize * action[inds, :]
    return samples


if __name__ == "__main__":
    print(obtainActionSamples(10, 0.1))
