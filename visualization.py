from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv("data.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    model = LinearRegression()
    model.fit(X, y)

    plt.scatter(y, model.predict(X))
    plt.xlabel("Actual Target")
    plt.ylabel("Predicted Target")
    plt.title("Actual vs Predicted Target")
    plt.show()

