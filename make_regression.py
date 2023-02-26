from sklearn.datasets import make_regression
import pandas as pd

n_features = 28
n_samples = 100000
n_informative = 25
noise = 0.5
bias = 10

if __name__ == '__main__':
    X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                           noise=noise, bias=bias, random_state=42)

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y

    df.to_csv("data.csv", index=False)

