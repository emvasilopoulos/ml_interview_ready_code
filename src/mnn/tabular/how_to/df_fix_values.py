import pandas as pd

if __name__ == "__main__":
    data = {
        "column_1": [-10.324, -9.33, -9.01, 0, 20, 33],
        "column_2": ["test1", "test2", "test33", "test3", "test5", "test1"],
    }
    df = pd.DataFrame(data)

    print("-- Before:\n", df)

    df["column_1"] = df["column_1"].apply(lambda x: -9 if x < -9 else x)
    df["column_2"] = df["column_2"].replace("test33", "test3")
    print("-- After:\n", df)
    pass
