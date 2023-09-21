from sklearn.model_selection import train_test_split
from src.tabular.synthetic_data import create_fake_table_dataframe

if __name__ == "__main__":
    synthetic_df = create_fake_table_dataframe(50, 10, 5, 0)
    X = synthetic_df.loc[:, synthetic_df.columns.str.startswith("feature")].values
    y = synthetic_df.loc[:, "label"].values

    n_samples = X.shape[0]
    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=int(n_samples * 0.3)
    )
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)
