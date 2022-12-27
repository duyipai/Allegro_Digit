import os

import cv2
import numpy as np
import pytact
import torch
from sklearn.model_selection import train_test_split

from digit.pose_estimation import Pose


class DynamicsModel(torch.nn.Module):
    def __init__(
        self,
        mean,
        std,
        hidden_size=32,
        dropout_p=0.1,
        activation=torch.nn.ReLU(),
    ):
        super().__init__()
        self.activation = activation
        self.mean = torch.from_numpy(mean).float().cuda()
        self.std = torch.from_numpy(std).float().cuda()
        self.mean.requires_grad = False
        self.std.requires_grad = False

        self.up1 = torch.nn.Linear(self.mean.shape[0] - 22, 22)
        self.up2 = torch.nn.Linear(22, 22)

        self.fc1 = torch.nn.Linear(22 * 2, hidden_size)  # state should have dim 22
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc4 = torch.nn.Linear(hidden_size, 4)
        self.drop = torch.nn.Dropout(p=dropout_p)

    def forward(self, state, action):
        action = (action - self.mean[22:]) / self.std[22:]
        action = self.activation(self.up1(action))
        action = self.up2(action)

        x = torch.cat([(state - self.mean[:22]) / self.std[:22], action], dim=1)
        x = self.activation(self.fc1(x))
        x = self.drop(x)
        x = self.activation(self.fc2(x))
        x = self.drop(x)
        x = self.activation(self.fc3(x))
        x = self.drop(x)
        x = self.fc4(x)
        return torch.cat(
            [x[:, 0:2], torch.atan(x[:, 2] / x[:, 3]).reshape(-1, 1) + np.pi / 2], dim=1
        )

    def setOutputNorm(self, normMultiplier):
        self.outputNorm = normMultiplier

    def predict(self, x):
        return self.forward(x) * self.outputNorm


def getPose(j, folder, lookupTable, reference, id):
    pose_estimator = Pose()
    pose_result = []
    for midname in ["_prev_Digit", "_Digit"]:
        img = cv2.imread(os.path.join(folder, str(j) + midname + str(id) + ".png"))
        depth = lookupTable(
            pytact.types.Frame(
                pytact.types.FrameEnc.DIFF, img.astype("float32") - reference
            )
        )
        depth.data[np.where(depth.data > 0.0)] = 0
        depth.data = -depth.data
        tmp = pose_estimator.get_pose(depth.data, img)
        pose_result.append(tmp)
        cv2.imwrite(
            os.path.join(folder, str(j) + midname + str(id) + "_pose.png"),
            pose_estimator.frame,
        )
        # print("Pose estimation done for " + str(j) + midname + str(id) + ".png")
    return pose_result


if __name__ == "__main__":

    folders = [os.path.join("data", "06-17:29:20"), os.path.join("data", "06-17:30:09")]
    lengths = [8, 4]
    n_epochs = 1000
    batch_size = 16
    model_path = "/home/yipai/Allegro_Digit/src/digit/data/digit_nn_normalized"
    model = pytact.models.Pixel2GradModel().cuda()
    model.load_state_dict(torch.load(model_path + ".pth", map_location="cuda"))
    model.eval()
    npz_file = np.load(model_path + ".npz")
    mean = npz_file["mean"]
    std = npz_file["std"]
    lookupTable = pytact.tasks.DepthFromLookup(
        model,
        mmpp=0.0487334006,
        scale=1.0,
        mean=mean,
        std=std,
        use_cuda=True,
        optimize=False,
    )

    mode = "!poseScan"
    if mode == "poseScan":
        for i in range(len(folders)):
            folder = folders[i]
            length = lengths[i]
            reference0 = cv2.imread(os.path.join(folder, "reference0.png")).astype(
                "float32"
            )
            reference1 = cv2.imread(os.path.join(folder, "reference1.png")).astype(
                "float32"
            )
            for j in range(length):
                if not os.path.exists(os.path.join(folder, str(j) + "_Digit0.png")):
                    continue
                getPose(j, folder, lookupTable, reference0, 0)
                getPose(j, folder, lookupTable, reference1, 1)
    else:
        input_data = []
        output_data = []
        for i in range(len(folders)):
            folder = folders[i]
            length = lengths[i]
            reference0 = cv2.imread(os.path.join(folder, "reference0.png")).astype(
                "float32"
            )
            reference1 = cv2.imread(os.path.join(folder, "reference1.png")).astype(
                "float32"
            )
            for j in range(length):
                if not os.path.exists(os.path.join(folder, str(j) + "_Digit0.png")):
                    continue
                poseTransition0 = getPose(j, folder, lookupTable, reference0, 0)
                poseTransition1 = getPose(j, folder, lookupTable, reference1, 1)
                npz_file = np.load(os.path.join(folder, str(j) + "_joints.npz"))
                prev_JointPose = np.append(
                    npz_file["prev_JointPose"][:4], npz_file["prev_JointPose"][-4:]
                )
                prev_JointTorque = np.append(
                    npz_file["prev_JointTorque"][:4], npz_file["prev_JointTorque"][-4:]
                )
                currentJointPose = np.append(
                    npz_file["currentJointPose"][:4], npz_file["currentJointPose"][-4:]
                )
                currentJointTorque = np.append(
                    npz_file["currentJointTorque"][:4],
                    npz_file["currentJointTorque"][-4:],
                )
                action = np.append(npz_file["action"][:4], npz_file["action"][-2:])
                input_data.append(
                    np.hstack(
                        (
                            prev_JointPose,
                            prev_JointTorque,
                            np.array(poseTransition0[0]),
                            np.array(poseTransition1[0]),
                            action,
                        )
                    ).reshape(1, -1)
                )
                output_data.append(np.array(poseTransition0[1]).reshape(1, -1))

        input_data = np.vstack(input_data)
        output_data = np.vstack(output_data)
        model = DynamicsModel(input_data.mean(axis=0), input_data.std(axis=0))
        model.setOutputNorm(
            torch.Tensor(
                (
                    np.hypot(reference0.shape[0], reference0.shape[1]),
                    np.pi,
                    np.hypot(reference1.shape[0], reference1.shape[1]),
                    np.pi,
                )
            )
        )
        model = model.cuda()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.MSELoss(reduction="mean")
        train_input, test_input, train_output, test_output = train_test_split(
            input_data, output_data, test_size=0.2, random_state=42
        )
        train_input = torch.from_numpy(train_input).float().cuda()
        train_output = torch.from_numpy(train_output).float().cuda()
        test_input = torch.from_numpy(test_input).float().cuda()
        test_output = torch.from_numpy(test_output).float().cuda()
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_input, train_output),
            batch_size=batch_size,
            shuffle=True,
        )

        for epoch in range(n_epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                output = model(batch[0][:, :22], batch[0][:, 22:])
                loss = loss_fn(output, batch[1])
                loss.backward()
                optimizer.step()
                print("Epoch: {}, Train Loss: {}".format(epoch, loss.item()))
            with torch.no_grad():
                model.eval()
                output = model(test_input[:, :22], test_input[:, 22:])
                loss = loss_fn(output, test_output)
                print("Epoch: {}, Test Loss: {}".format(epoch, loss.item()))
        torch.save(model, os.path.join("data", "dynamics_model.pth"))
