import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from expanding_window_engineering import *


df = pd.read_csv("cleaned_sp500.csv")
df["Last Month Price"] = df["SP500"].shift(1)
df["Growth Factor"] = (df["SP500"] + df["Dividend"] / 12) / df["Last Month Price"]
df.loc[0, "Growth Factor"] = 1
df["Cumulative Growth"] = df["Growth Factor"].cumprod()
df["Real Cumulative Growth"] = (df["Cumulative Growth"] * (df["Consumer Price Index"].iloc[-1] / df["Consumer Price Index"]))
df["Future Value"] = df["Real Cumulative Growth"].shift(-120)
df["Annual Return"] = ((df["Future Value"] / df["Real Cumulative Growth"]) ** (1 / 10)) - 1
model_df = df[df["Annual Return"].notna()].reset_index(drop=True)

features = model_df[["PE10"]]
label = model_df["Annual Return"]

window_size = 600
cutoff = 600
step = 60







def rolling_window_LR(feature):

    cutoff = 600

    mse_scores = []
    baseline_mse_scores = []
    times = []

    beginning = 0

    while cutoff + 60 <= len(features):

        X_train = model_df[feature][beginning:cutoff]
        y_train = label[beginning:cutoff]

        X_test = model_df[feature][cutoff:cutoff+60]
        y_test = label[cutoff:cutoff+60]

        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train)

        predictions = lin_reg.predict(X_test)

        mse_scores.append(mean_squared_error(y_test, predictions))

        baseline_prediction = [y_train.mean()] * len(y_test)
        baseline_mse_scores.append(mean_squared_error(y_test, baseline_prediction))

        times.append(model_df["Date"][cutoff][:4])

        cutoff += 60
        beginning+=60

    X_train = model_df[feature][beginning:cutoff]
    y_train = label[beginning:cutoff]

    X_test = model_df[feature][cutoff:]
    y_test = label[cutoff:]

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    predictions = lin_reg.predict(X_test)

    mse_scores.append(mean_squared_error(y_test, predictions))

    baseline_prediction = [y_train.mean()] * len(y_test)
    baseline_mse_scores.append(mean_squared_error(y_test, baseline_prediction))

    times.append(model_df["Date"][cutoff][:4])

    return baseline_mse_scores, mse_scores, times

rol_base_mse, rol_lr_mse, times= rolling_window_LR(["PE10"])
rol_knn_mse = []
rol_kernel_mse = []

baseline_mse, exp_lr_mse, times = expanding_window_LR(["PE10"])
exp_knn_mse = expanding_window_KNN(["PE10"])
exp_kernel_mse = kernel_reg(["PE10"])

lr_pe10_expanding = oos_r2(exp_lr_mse, baseline_mse)

lr_pe10_rolling = oos_r2(rol_lr_mse, rol_base_mse)


