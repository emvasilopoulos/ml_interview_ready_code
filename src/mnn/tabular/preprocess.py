from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    OneHotEncoder,
    LabelEncoder,
    KBinsDiscretizer,
)

# scaler = StandardScaler()
# scaler.fit(X)
# scaled_x = scaler.transform(X)


"""
- Method: MinMaxScaler
- Reason: Scales features to a specific range (typically [0, 1]) to ensure that all features have the same scale.
This is useful for algorithms that are sensitive to feature scaling, such as K-Means and Support Vector Machines (SVMs).

- Method: StandardScaler
- Reason: Transforms data to have a mean of 0 and a standard deviation of 1. 
It's useful when your data contains outliers and 
is required by many machine learning algorithms like Gradient Descent-based models
 (e.g., Linear Regression, Logistic Regression) to ensure faster convergence

- Method: RobustScaler
- Reason: Similar to standardization but robust to outliers. 
It scales data based on the median and the interquartile range (IQR),
making it less affected by extreme values. Useful when your dataset has outliers that can distort standardization.

- Method: OneHotEncoder or get_dummies function
- Reason: Converts categorical variables into binary vectors (0s and 1s) for algorithms
that can't directly handle categorical data.
It allows you to represent categorical features in a way that the model can understand.

- Method: LabelEncoder
- Reason: Converts categorical labels into numeric labels. 
Useful when dealing with ordinal categorical data (categories with a specific order),
which some models can work with directly.

- Method: LabelEncoder
- Reason: Converts categorical labels into numeric labels.
Useful when dealing with ordinal categorical data (categories with a specific order),
which some models can work with directly.
"""
