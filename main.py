import pandas as pd
from sklearn.model_selection import train_test_split

# getting data from csv
df = pd.read_csv("data/Heart_disease_statlog.csv")
print(df.head()) # show 5 rows
print(df.info()) # show column types
print(df.describe()) # show stats, numeric columns
print(df["target"].value_counts()) # show the label/target counts

# remote the target column from dataframe
# axis=1, drop a column (axis=0, drop a row)
x = df.drop("target", axis=1)

# get the target column only
y = df["target"]

# x featuers, y label
# x_train features for training, y_train labels matching x_train
# x_test features for testing, y_test for matching x_test
# test_size=0.2 (20% for testing, 80% for training)
# random_state=42 (seed number for shuffle, 42 is a common convention number)
# use split for simulating real-world unseen data, giving honest accuracy score.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(x_train.shape, x_test.shape)