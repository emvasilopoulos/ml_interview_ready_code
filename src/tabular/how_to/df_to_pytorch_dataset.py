import pandas as pd
from sklearn.model_selection import train_test_split
from src.tabular.dataset import TableDataset
from torch.utils.data import DataLoader


if __name__ == "__main__":
    synthetic_df = pd.read_csv("src/tabular/data/synthetic_table.csv")
    X = synthetic_df.loc[:, synthetic_df.columns.str.startswith("feature")].values
    y = synthetic_df.loc[:, "label"].values

    n_samples = X.shape[0]
    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=int(n_samples * 0.3)
    )

    train_dataset = TableDataset(train_x, train_y)
    test_dataset = TableDataset(test_x, test_y)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    print("---------- train")
    for x, y in train_loader:
        print(x.shape)
        print(y.shape)

    print("---------- test")
    for x, y in test_loader:
        print(x.shape)
        print(y.shape)
