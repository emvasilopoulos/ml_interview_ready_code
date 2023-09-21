import random
import pandas as pd
from sklearn.datasets import make_classification


def create_fake_table_dataframe(
    n_samples: int,
    n_features: int,
    n_classes: int,
    n_repeated: int = 0,
) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_repeated=n_repeated,
        n_clusters_per_class=1,
        n_informative=5,
        scale=[
            random.choice([random.randint(50, 1000), random.randint(1, 5)])
            for _ in range(n_features)
        ],
    )

    dataset = {}
    for i in range(n_features):
        dataset[f"feature_{i+1}"] = X[:, i]
    dataset["label"] = y
    return pd.DataFrame(dataset)


if __name__ == "__main__":
    create_fake_table_dataframe(10000, 10, 5, 0).to_csv(
        "ten_thousand_samples.csv", index=False
    )
