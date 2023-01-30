import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if __name__ == "__main__":
    data = []
    runs = 30
    for i in range(runs):
        data.append(np.load("data/poseRegErrorsOpt" + str(i) + ".npy"))
    optData = np.stack(data, axis=0) * 0.048733400
    optMean = np.mean(optData, axis=0)
    optStd = np.std(optData, axis=0)
    for i in range(runs):
        data.append(np.load("data/poseRegErrorsSamp" + str(i) + ".npy"))
    sampData = np.stack(data, axis=0) * 0.0487334006
    sampMean = np.mean(sampData, axis=0)
    sampStd = np.std(sampData, axis=0)
    x = np.arange(sampMean.shape[0])
    sns.set()
    ftsise = 15

    (line_1,) = plt.plot(x, sampMean[:, 1], "b-")  # actual loss
    lower = sampMean[:, 1] - sampStd[:, 1]
    lower[lower < 0] = 0
    fill_1 = plt.fill_between(
        x,
        lower,
        sampMean[:, 1] + sampStd[:, 1],
        color="b",
        alpha=0.2,
    )
    (line_2,) = plt.plot(x[1:], sampMean[1:, 0], "r--")
    lower = sampMean[1:, 0] - sampStd[1:, 0]
    lower[lower < 0] = 0
    fill_2 = plt.fill_between(
        x[1:],
        lower,
        sampMean[1:, 0] + sampStd[1:, 0],
        color="r",
        alpha=0.2,
    )
    plt.margins(x=0)

    plt.legend(
        [(line_1, fill_1), (line_2, fill_2)],
        ["Actual Distance", "Predicted Distance"],
        fontsize=ftsise,
    )
    plt.title("Without Optimization", fontsize=ftsise)
    plt.xlabel("Action Step", fontsize=ftsise)
    plt.ylabel("Distance to Center/mm", fontsize=ftsise)
    plt.savefig("data/poseRegErrorsSamp.png")
    plt.close("all")

    plt.figure()
    (line_1,) = plt.plot(x, optMean[:, 1], "b-")  # actual loss
    lower = optMean[:, 1] - optStd[:, 1]
    lower[lower < 0] = 0
    fill_1 = plt.fill_between(
        x,
        lower,
        optMean[:, 1] + optStd[:, 1],
        color="b",
        alpha=0.2,
    )
    (line_2,) = plt.plot(x[1:], optMean[1:, 0], "r--")
    lower = optMean[1:, 0] - optStd[1:, 0]
    lower[lower < 0] = 0
    fill_2 = plt.fill_between(
        x[1:],
        lower,
        optMean[1:, 0] + optStd[1:, 0],
        color="r",
        alpha=0.2,
    )
    plt.margins(x=0)

    plt.legend(
        [(line_1, fill_1), (line_2, fill_2)],
        ["Actual Distance", "Predicted Distance"],
        fontsize=ftsise,
    )
    plt.title("With Optimization", fontsize=ftsise)
    plt.xlabel("Action Step", fontsize=ftsise)
    plt.ylabel("Distance to Center/mm", fontsize=ftsise)
    plt.savefig("data/poseRegErrorsOpt.png")
    plt.close("all")
