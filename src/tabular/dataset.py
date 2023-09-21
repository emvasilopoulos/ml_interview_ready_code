from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class TableDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Constructor of TableDataset class
        Creates tensors with dtype=torch.float32 for both X (features) and y (labels)

        Args:
            X (np.ndarray): numpy array of shape (n_samples, n_features)
            y (np.ndarray): numpy array of shape (n_samples, 1)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index: Any) -> Any:
        return self.X[index], self.y[index]


if __name__ == "__main__":
    from .synthetic_data import create_fake_table_dataframe

    synthetic_df = create_fake_table_dataframe(10, 5, 5, 0)
    X = synthetic_df.loc[:, synthetic_df.columns.str.startswith("feature")].values
    y = synthetic_df.loc[:, "label"].values

    print(train_test_split(X, y, test_size=30))
