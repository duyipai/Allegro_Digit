import glob
import os
import random
import time

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from digit.pose_estimation import Pose

torch.manual_seed(int(((time.time() % 1000.0) * 1000000) % (2**31)))
random.seed(int(((time.time() % 1000.0) * 1000000) % (2**31)))
np.random.seed(int(((time.time() % 1000.0) * 1000000) % (2**31)))
torch.backends.cudnn.benchmark = True


class DynamicsModel(torch.nn.Module):
    def __init__(
        self,
        mean,
        std,
        withAngle,
        withTorque,
        hidden_size=32,
        dropout_p=0.1,
        activation=torch.nn.ReLU(),
    ):
        super().__init__()
        self.activation = activation
        self.withAngle = withAngle
        self.withTorque = withTorque
        self.mean = torch.from_numpy(mean).float().cuda()
        self.std = torch.from_numpy(std).float().cuda()
        self.mean.requires_grad = False
        self.std.requires_grad = False

        if self.withTorque:
            up_to = 16
        else:
            up_to = 8
        self.up1 = torch.nn.Linear(7, up_to)
        self.up2 = torch.nn.Linear(up_to, up_to)

        self.up3 = torch.nn.Linear(4, up_to)
        self.up4 = torch.nn.Linear(up_to, up_to)

        if self.withTorque:
            self.fc1 = torch.nn.Linear(16 * 3, hidden_size)  # hand state 16 *3
        else:
            self.fc1 = torch.nn.Linear(8 * 3, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc4 = torch.nn.Linear(hidden_size, hidden_size)
        if self.withAngle:
            self.fc5 = torch.nn.Linear(hidden_size, 5)
        else:
            self.fc5 = torch.nn.Linear(hidden_size, 3)
        self.drop = torch.nn.Dropout(p=dropout_p)

    def forward(self, hand_state, tactile_state, action):
        if self.withTorque:
            action = (action - self.mean[20:]) / self.std[20:]
            tactile_state = (tactile_state - self.mean[16:20]) / self.std[16:20]
        else:
            action = (action - self.mean[12:]) / self.std[12:]
            tactile_state = (tactile_state - self.mean[8:12]) / self.std[8:12]

        action = self.activation(self.up1(action))
        action = self.up2(action)

        tactile_state = self.activation(self.up3(tactile_state))
        tactile_state = self.up4(tactile_state)

        if self.withTorque:
            x = torch.cat(
                [(hand_state - self.mean[:16]) / self.std[:16], tactile_state, action],
                dim=1,
            )  # 16*3
        else:
            x = torch.cat(
                [(hand_state - self.mean[:8]) / self.std[:8], tactile_state, action],
                dim=1,
            )  # 16*3
        x = self.activation(self.fc1(x))
        x = self.drop(x)
        x = self.activation(self.fc2(x))
        x = self.drop(x)
        x = self.activation(self.fc3(x))
        x = self.drop(x)
        x = self.fc4(x)
        x = self.drop(x)
        x = self.fc5(x)
        if self.withAngle:
            return torch.cat(
                [
                    x[:, 0:3],
                    (torch.atan(x[:, 3] / x[:, 4]).reshape(-1, 1) + np.pi / 2) / np.pi,
                ],
                dim=1,
            )
        else:
            return x

    def setOutputNorm(self, normMultiplier):
        self.contactNorm = normMultiplier
        self.contactNorm[1:] = 1.0

    def predict(self, hand_state, tactile_state, action):
        self.eval()
        with torch.no_grad():
            ret = self.forward(hand_state, tactile_state, action)
        return ret

    def loss_fn(self, pred, target, reduce=True):
        failed_ind = torch.logical_and(
            torch.abs(target[:, 1] - 0.5) < 1e-5,
            torch.abs(target[:, 2] - 0.5) < 1e-5,
        )
        success_ind = torch.logical_not(failed_ind)
        failed_loss = torch.abs(target[failed_ind, 0] - pred[failed_ind, 0]).reshape(
            -1, 1
        )
        success_loss1 = torch.abs(target[success_ind, :-1] - pred[success_ind, :-1])
        success_loss2 = (target[success_ind, -1] - pred[success_ind, -1]).reshape(
            -1, 1
        )  # loss on angle

        success_loss2 = torch.min(
            torch.hstack(
                (
                    torch.abs(success_loss2 - 1.0),
                    torch.abs(success_loss2 + 1.0),
                    torch.abs(success_loss2),
                )
            ),
            dim=1,
        )[0]
        if reduce:  # used during training
            lo = (failed_loss.sum() + success_loss1.sum() + success_loss2.sum()) / (
                failed_loss.shape[0]
                + success_loss1.shape[0] * success_loss1.shape[1]
                + success_loss2.shape[0]
            )
            # if torch.isnan(lo).sum() > 0:
            #     print("NAN")
            return lo
        else:
            return np.hstack(
                (
                    np.array(failed_loss.mean().item()),
                    success_loss1.mean(axis=0).cpu().numpy(),
                    np.array(success_loss2.mean().item()),
                )
            ).reshape((1, -1))


def getPose(j, folder, reference, id):
    pose_estimator = Pose()
    pose_result = []
    for midname in ["_prev_Digit", "_Digit"]:
        img = cv2.imread(os.path.join(folder, str(j) + midname + str(id) + ".png"))
        diff = img.astype("float32") - reference
        tmp = pose_estimator.get_pose(diff, img)
        pose_result.append(tmp)
        cv2.imwrite(
            os.path.join(folder, str(j) + midname + str(id) + "Pose.png"),
            pose_estimator.frame,
        )
        # print("Pose estimation done for " + str(j) + midname + str(id) + ".png")
    return pose_result


if __name__ == "__main__":

    parent_folder = os.path.join("data", "use")
    fs = os.listdir(parent_folder)
    folders = []
    lengths = []
    for f in fs:
        folders.append(os.path.join(parent_folder, f))
        files = glob.glob(os.path.join(parent_folder, f, "*.png"))
        max_length = 0
        for file in files:
            pureFileName = file.split("/")[-1]
            if pureFileName == "reference0.png":
                continue
            else:
                ind = int(pureFileName.split("_")[0])
                if ind > max_length:
                    max_length = ind
        lengths.append(max_length + 1)

    n_epochs = 200
    batch_size = 32
    num_of_models = 16
    withAngle = True
    withTorque = False
    model_id = 0
    errors = []
    print("Start training with angle ", withAngle, " and torque ", withTorque)

    while model_id < num_of_models:
        input_data = []
        output_data = []
        for i in range(len(folders)):
            folder = folders[i]
            length = lengths[i]
            reference0 = cv2.imread(os.path.join(folder, "reference0.png")).astype(
                "float32"
            )
            for j in range(length):
                if not os.path.exists(os.path.join(folder, str(j) + "_Digit0.png")):
                    continue
                poseTransition0 = getPose(j, folder, reference0, 0)
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
                action = np.append(npz_file["action"][:4], npz_file["action"][-3:])
                if withTorque:
                    singleIn = np.hstack(
                        (
                            prev_JointPose,  # 8
                            prev_JointTorque,  # 8
                            np.array(poseTransition0[0]),  # 4
                            action,  # 7
                        )
                    ).reshape(1, -1)
                else:
                    singleIn = np.hstack(
                        (
                            prev_JointPose,  # 8
                            np.array(poseTransition0[0]),  # 4
                            action,  # 7
                        )
                    ).reshape(1, -1)
                if withAngle:
                    singleOut = np.array(poseTransition0[1]).reshape(1, -1)
                else:
                    singleOut = np.array(poseTransition0[1][:-1]).reshape(1, -1)
                input_data.append(singleIn)
                output_data.append(singleOut)

        input_data = np.vstack(input_data)
        output_data = np.vstack(output_data)
        ind = np.where(output_data[:, 0] > 5.0)
        input_data = np.delete(input_data, ind, axis=0)
        output_data = np.delete(output_data, ind, axis=0)
        if withTorque:
            contacts = input_data[:, 16]
        else:
            contacts = input_data[:, 8]
        import matplotlib.pyplot as plt

        plt.figure()
        plt.hist(contacts, bins=100)
        plt.savefig("input_contact.png")
        plt.figure()
        plt.hist(output_data[:, 0], bins=100)
        plt.savefig("output_contact.png")
        plt.close("all")
        model = DynamicsModel(
            input_data.mean(axis=0), input_data.std(axis=0), withAngle, withTorque
        )
        model.setOutputNorm(
            torch.from_numpy(output_data.max(axis=0)).float(),
        )
        # print("Output Norm and data shape: ", model.contactNorm, output_data.shape)
        model = model.cuda()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        output_data[:, 0] = output_data[:, 0] / output_data.max(axis=0)[0]

        train_input, test_input, train_output, test_output = train_test_split(
            input_data,
            output_data,
            test_size=0.2,
            random_state=np.random.randint(2**31),
        )
        # augment data for rotation
        t_shape = train_input.shape[0]
        train_input = np.tile(train_input, (3, 1))
        if withTorque:
            train_input[t_shape : (t_shape * 2), 19] -= 1.0
            train_input[(2 * t_shape) :, 19] += 1.0
        else:
            train_input[t_shape : (t_shape * 2), 11] -= 1.0
            train_input[(2 * t_shape) :, 11] += 1.0
        train_output = np.tile(train_output, (3, 1))
        # end augmentation
        train_input = torch.from_numpy(train_input).float().cuda()
        train_output = torch.from_numpy(train_output).float().cuda()
        test_input = torch.from_numpy(test_input).float().cuda()
        test_output = torch.from_numpy(test_output).float().cuda()
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_input, train_output),
            batch_size=batch_size,
            shuffle=True,
        )

        found_nan = False
        for epoch in range(n_epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                if withTorque:
                    output = model(
                        batch[0][:, :16], batch[0][:, 16:20], batch[0][:, 20:]
                    )
                else:
                    output = model(batch[0][:, :8], batch[0][:, 8:12], batch[0][:, 12:])
                loss = model.loss_fn(output, batch[1])
                loss.backward()
                optimizer.step()
                # print("Epoch: {}, Train Loss: {}".format(epoch, loss.item()))
            with torch.no_grad():
                model.eval()
                if withTorque:
                    output = model(
                        test_input[:, :16], test_input[:, 16:20], test_input[:, 20:]
                    )
                else:
                    output = model(
                        test_input[:, :8], test_input[:, 8:12], test_input[:, 12:]
                    )
                loss = model.loss_fn(output, test_output)
                # print("Epoch: {}, Test Loss: {}".format(epoch, loss.item()))
                if torch.isnan(loss).sum().item() > 0:
                    found_nan = True
                    print("Found nan, break")
                    break
        if found_nan:
            continue
        with torch.no_grad():
            model.eval()
            if withTorque:
                output = model.predict(
                    test_input[:, :16], test_input[:, 16:20], test_input[:, 20:]
                )
            else:
                output = model.predict(
                    test_input[:, :8], test_input[:, 8:12], test_input[:, 12:]
                )
            error = model.loss_fn(output, test_output, False)
            error[0, 0] *= model.contactNorm[0].item()
            error[0, 1] *= model.contactNorm[0].item()
            print("Error is ", error)
            errors.append(error)
        torch.save(
            model, os.path.join("data", "noTdynamics_model" + str(model_id) + ".pth")
        )
        print("Model" + str(model_id) + " saved")
        model_id += 1
    errors = np.vstack(errors)
    np.save(os.path.join("data", "errors.npy"), errors)
    print(errors)
