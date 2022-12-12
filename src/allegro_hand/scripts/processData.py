import os
from math import acos, pi

import cv2
import numpy as np
import pytact
import torch
from numpy import linalg as LA
from sklearn.model_selection import train_test_split


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

        self.fc1 = torch.nn.Linear(self.mean.shape[0], hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc4 = torch.nn.Linear(hidden_size, 4)
        self.drop = torch.nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.activation(self.fc1(x))
        x = self.drop(x)
        x = self.activation(self.fc2(x))
        x = self.drop(x)
        x = self.activation(self.fc3(x))
        x = self.drop(x)
        x = self.fc4(x)
        return torch.cat(
            [x[:, 0:2], torch.atan(x[:, 2] / x[:, 3]).reshape(-1, 1) + pi / 2], dim=1
        )

    def setOutputNorm(self, normMultiplier):
        self.outputNorm = normMultiplier

    def predict(self, x):
        return self.forward(x) * self.outputNorm


class Pose:
    def __init__(self):
        self.pose = None
        self.area = 0
        self.depth = None

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

    def get_pose(self, depth, frame):
        self.depth = depth.copy()
        thresh = max(0.00003, depth.max() / 2)

        mask = depth > thresh
        # Display the resulting frame
        coors = np.where(mask == 1)
        X = coors[1].reshape(-1, 1)
        y = coors[0].reshape(-1, 1)

        self.area = 0
        if len(X) > 10:
            pts = np.concatenate([X, y], axis=1)
            v_max, v_min, w_max, w_min = self.PCA(pts)
            if v_max[1] < 0:
                v_max *= -1
            m = np.mean(pts, 0).reshape(-1)

            # Record pose estimation
            self.pose = (v_max, v_min, w_max, w_min, m)
            self.area = len(X)

            self.draw_ellipse(frame, self.pose)
            rho = (
                (v_max[0] * m[1] - v_max[1] * m[0])
                / (v_max[0] ** 2 + v_max[1] ** 2) ** 0.5
            ).item()
            theta = acos(v_max[0] / (v_max[0] ** 2 + v_max[1] ** 2) ** 0.5)
            return (
                depth.mean() / 0.3,
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
                output = model(batch[0])
                loss = loss_fn(output, batch[1])
                loss.backward()
                optimizer.step()
                print("Epoch: {}, Train Loss: {}".format(epoch, loss.item()))
            with torch.no_grad():
                model.eval()
                output = model(test_input)
                loss = loss_fn(output, test_output)
                print("Epoch: {}, Test Loss: {}".format(epoch, loss.item()))
        torch.save(model, os.path.join("data", "dynamics_model.pth"))
