import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns


def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes).to(labels.device)
    return y[labels]


class ViewDataMO(Dataset):

    def __init__(self, root, train, view_list):
        super(ViewDataMO, self).__init__()
        train = "tr" if train else "te"
        self.X = dict()
        for idx, v_num in enumerate(view_list):
            self.X[idx] = pd.read_csv(f"{root}/{v_num}_{train}.csv").to_numpy()

        y = pd.read_csv(f"{root}/labels_{train}.csv").to_numpy()

        tmp = np.zeros(y.shape[0])
        y = np.reshape(y, np.shape(tmp))
        self.y = y

        self.is_single_view = len(view_list) == 1

    def __getitem__(self, index):
        if self.is_single_view:
            data = self.X[0][index].astype(np.float32)
        else:
            data = {v_num: self.X[v_num][index].astype(np.float32) for v_num in range(len(self.X))}

        target = self.y[index]
        return data, target

    def __len__(self):
        return self.X[0].shape[0]


def plot_uncertainty(preds, us, stage):
    preds = np.array(preds)
    us = np.array(us)

    unique_classes = np.unique(preds)
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    palette = sns.color_palette("tab10", len(unique_classes))

    for i, value in enumerate(unique_classes):
        mask = (preds == value)
        sns.histplot(
            us[mask],
            bins=30,
            label=f'Class {value}',
            color=palette[i],
            kde=False,
            element="step",
            alpha=0.5
        )

    plt.title(f'Uncertainty Distribution for stage {stage}', fontsize=16)
    plt.xlabel('Uncertainty', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend(title='Class', fontsize=12)
    plt.grid(True)
    plt.show()
