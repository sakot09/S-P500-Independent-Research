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

features = model_df[["PE10", "Long Interest Rate"]]
label = model_df["Annual Return"]

def expanding_window_LR(feature):

    cutoff = 600

    mse_scores = []
    baseline_mse_scores = []
    times = []

    while cutoff + 60 <= len(features):

        X_train = model_df[feature][:cutoff]
        y_train = label[:cutoff]

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

    X_train = model_df[feature][:cutoff]
    y_train = label[:cutoff]

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

def expanding_window_KNN(feature):
    
    
    
    neighbors  = 105
    cutoff = 600

    mse_scores = []
    baseline_mse_scores = []
    times = []

    while cutoff + 60 <= len(features):

        X_train = model_df[feature][:cutoff]
        y_train = label[:cutoff]

        X_test = model_df[feature][cutoff:cutoff+60]
        y_test = label[cutoff:cutoff+60]

        knn = KNeighborsRegressor(neighbors)
        knn.fit(X_train, y_train)

        predictions = knn.predict(X_test)

        mse_scores.append(mean_squared_error(y_test, predictions))

        baseline_prediction = [y_train.mean()] * len(y_test)
        baseline_mse_scores.append(mean_squared_error(y_test, baseline_prediction))

        times.append(model_df["Date"][cutoff][:4])

        cutoff += 60

    X_train = model_df[feature][:cutoff]
    y_train = label[:cutoff]

    X_test = model_df[feature][cutoff:]
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

def kernel_reg(feature):
    def gaussian_kernel(distance, h):
        weight = np.exp(-0.5 * ((distance/h)**2))
        return weight

    def predict_one(test, training_PE10, training_returns, h):
        distances = []

        for value in training_PE10:
            distances.append(value - test)

        distances = np.array(distances)

        weights = gaussian_kernel(distances, h)

        sum_weights = sum(weights)

        sum_mult = sum(training_returns * weights)


        return sum_mult/sum_weights

    def bandwidth_sel(training_PE10, training_returns, h_values):
        best_h = None

        training_returns = np.array(training_returns)
        training_PE10 = np.array(training_PE10)

        best_error = float('inf')

        for h in h_values:

            errors = []

            for i in range(len(training_PE10)):

                pe10 = np.delete(training_PE10, i)

                returns = np.delete(training_returns, i)

                pred = predict_one(training_PE10[i], pe10, returns, h)

                error = (pred - training_returns[i])**2

                errors.append(error)

            avg_error = np.mean(errors)

            if avg_error < best_error:
                best_error = avg_error
                best_h = h

        return best_h

    cutoff = 600
    training_PE10 = np.array(model_df[feature][:cutoff])

    training_returns = np.array(label[:cutoff])

    h_values = np.arange(1,30,1)

    best_h = bandwidth_sel(training_PE10, training_returns, h_values)

    times = []

    predictions = []

    mse_scores = []

    while cutoff + 60 <= len(features):
    
        training_PE10 = np.array(model_df[feature][:cutoff])
        training_returns = np.array(label[:cutoff])

        X_test = np.array(model_df[feature][cutoff: cutoff+60])
        y_test = np.array(label[cutoff:cutoff+60])


        predictions = []

        for i in range(len(X_test)):

            pred = predict_one(X_test[i], training_PE10, training_returns, best_h)
            predictions.append(pred)

        mse_scores.append(mean_squared_error(y_test, predictions))

        times.append(model_df["Date"][cutoff][:4])

        cutoff += 60

    predictions = []

    training_PE10 = np.array(model_df[feature][:cutoff])
    training_returns = np.array(label[:cutoff])

    X_test = np.array(model_df[feature][cutoff:])
    y_test = np.array(label[cutoff:])

    for i in range(len(X_test)):

        pred = predict_one(X_test[i], training_PE10, training_returns, best_h)
        predictions.append(pred)

    mse_scores.append(mean_squared_error(y_test, predictions))

    return mse_scores







    

knn_mse = expanding_window_KNN()

kernel_mse = kernel_reg()

baseline_mse, lr_pe10_mse, times = expanding_window_LR(["PE10"])
_, lr_rate_mse, _ = expanding_window_LR(["Long Interest Rate"])
_, lr_combined_mse, _ = expanding_window_LR(["PE10", "Long Interest Rate"])


