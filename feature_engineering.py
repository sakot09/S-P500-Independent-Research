import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor
import statsmodels.api as sm
import numpy as np



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

def expanding_window_LR():

    cutoff = 600

    mse_scores = []
    baseline_mse_scores = []
    times = []

    while cutoff + 60 <= len(features):

        X_train = features[:cutoff]
        y_train = label[:cutoff]

        X_test = features[cutoff:cutoff+60]
        y_test = label[cutoff:cutoff+60]

        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train)

        predictions = lin_reg.predict(X_test)

        mse_scores.append(mean_squared_error(y_test, predictions))

        baseline_prediction = [y_train.mean()] * len(y_test)
        baseline_mse_scores.append(mean_squared_error(y_test, baseline_prediction))

        times.append(model_df["Date"][cutoff][:4])

        cutoff += 60

    X_train = features[:cutoff]
    y_train = label[:cutoff]

    X_test = features[cutoff:]
    y_test = label[cutoff:]

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    predictions = lin_reg.predict(X_test)

    mse_scores.append(mean_squared_error(y_test, predictions))

    baseline_prediction = [y_train.mean()] * len(y_test)
    baseline_mse_scores.append(mean_squared_error(y_test, baseline_prediction))

    times.append(model_df["Date"][cutoff][:4])

    plt.figure(figsize=(9,5))

    knn_mse = expanding_window_KNN()

    plt.plot(times, mse_scores, label = "Linear Regression")
    plt.plot(times, knn_mse, label = "KNN Regresion")
    plt.plot(times, baseline_mse_scores, label = "Historical Mean")

    plt.xlabel("Test Period")
    plt.ylabel("Mean Squared Error")
    plt.title("Expanding-Window Forecast Performance Between Models")
    plt.xticks(fontsize=8)
    plt.legend()

    plt.tight_layout()
    plt.savefig("model_comparison.png")

    """model_mse_avg = statistics.mean(mse_scores)
    baseline_mse_avg = statistics.mean(baseline_mse_scores)
    improvement = (baseline_mse_avg - model_mse_avg)/baseline_mse_avg

    print(model_mse_avg)
    print(baseline_mse_avg)
    print(improvement)"""

def model_accuracy():
    cutoff = int(.8 * len(features))

    X_train = features[:cutoff]
    y_train = label[:cutoff]

    X_test = features[cutoff:]
    y_test = label[cutoff:]

    linreg = LinearRegression()
    linreg.fit(X_train, y_train)

    predictions = linreg.predict(X_test)

    plt.figure(figsize=(9,5))
    plt.scatter(y_test, predictions)
    plt.plot([-.1,.2], [-.1,.2], "--k", label = "Correct Prediction")

    plt.xlabel("True Return")
    plt.ylabel("Predicted Return")
    plt.title("Model Accuracy")

    plt.savefig("model_accuracy.png")

def lin_reg_appropriate():

    X = features
    y = label

    reg = LinearRegression()

    reg.fit(X,y)

    preds = reg.predict(X)

    residuals = y - preds

    plt.figure(figsize=(6,6))

    stats.probplot(residuals, dist="norm", plot=plt)

    plt.title("Normal Q-Q Plot of Residuals")

    plt.tight_layout()
    plt.savefig("qqresidplot.png")

def expanding_window_KNN():
    
    
    
    neighbors  = 105
    cutoff = 600

    mse_scores = []
    baseline_mse_scores = []
    times = []

    while cutoff + 60 <= len(features):

        X_train = features[:cutoff]
        y_train = label[:cutoff]

        X_test = features[cutoff:cutoff+60]
        y_test = label[cutoff:cutoff+60]

        knn = KNeighborsRegressor(neighbors)
        knn.fit(X_train, y_train)

        predictions = knn.predict(X_test)

        mse_scores.append(mean_squared_error(y_test, predictions))

        baseline_prediction = [y_train.mean()] * len(y_test)
        baseline_mse_scores.append(mean_squared_error(y_test, baseline_prediction))

        times.append(model_df["Date"][cutoff][:4])

        cutoff += 60

    X_train = features[:cutoff]
    y_train = label[:cutoff]

    X_test = features[cutoff:]
    y_test = label[cutoff:]

    knn = KNeighborsRegressor(neighbors)
    knn.fit(X_train, y_train)

    predictions = knn.predict(X_test)

    mse_scores.append(mean_squared_error(y_test, predictions))

    baseline_prediction = [y_train.mean()] * len(y_test)
    baseline_mse_scores.append(mean_squared_error(y_test, baseline_prediction))

    times.append(model_df["Date"][cutoff][:4])

    return mse_scores


   

def newey_west_corr():
    x = features

    x = sm.add_constant(x)

    y = label

    z = sm.OLS(y, x)

    result = z.fit()

    print(result.summary())

    result = z.fit(cov_type = 'HAC', cov_kwds={"maxlags":120})

    print(result.summary())

def kernel_reg():
    def gaussian_kernel(distance, h):
        weight = np.exp(-0.5 * ((distance/h)**2))
        return weight

    def 

newey_west_corr()
