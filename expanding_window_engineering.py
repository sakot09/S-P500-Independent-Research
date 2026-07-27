import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor
import statsmodels.api as sm
import numpy as np
from sklearn.preprocessing import StandardScaler



df = pd.read_csv("cleaned_sp500.csv")

df["Last Month Price"] = df["SP500"].shift(1)

df["Growth Factor"] = (df["SP500"] + df["Dividend"] / 12) / df["Last Month Price"]

df.loc[0, "Growth Factor"] = 1

df["Cumulative Growth"] = df["Growth Factor"].cumprod()

df["Real Cumulative Growth"] = (df["Cumulative Growth"] * (df["Consumer Price Index"].iloc[-1] / df["Consumer Price Index"]))

df["Future Value"] = df["Real Cumulative Growth"].shift(-120)

df["Annual Return"] = ((df["Future Value"] / df["Real Cumulative Growth"]) ** (1 / 10)) - 1

df["Dividend Yield"] = df["Dividend"] / df["SP500"]

model_df = df[df["Annual Return"].notna()].reset_index(drop=True)

features = model_df[["PE10", "Dividend Yield"]]
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

def select_k(feature, k_values):
    cutoff = int(0.8 * len(model_df))
    
    X_train = model_df[feature][:cutoff]
    y_train = label[:cutoff]
    
    X_test = model_df[feature][cutoff:]
    y_test = label[cutoff:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    best_error = float('inf')
    mse_per_k = []
    
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        predictions = knn.predict(X_test_scaled)
        mse = mean_squared_error(y_test, predictions)
        mse_per_k.append(mse)
        
        if mse < best_error:
            best_error = mse
    
    threshold = 1e-4
    optimal_k = k_values[-1]

    for i in range(1, len(mse_per_k)):
        pct_change = (mse_per_k[i-1] - mse_per_k[i]) / mse_per_k[i-1]
        if 0 < pct_change < threshold:
            optimal_k = k_values[i-1]
            break

    return optimal_k

    

def expanding_window_KNN(feature):
    
    
    
    neighbors  = 223
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

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)

        X_test_scaled = scaler.transform(X_test)

        knn.fit(X_train_scaled, y_train)

        predictions = knn.predict(X_test_scaled)

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

    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    knn.fit(X_train_scaled, y_train)

    predictions = knn.predict(X_test_scaled)

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

def kernel_reg(features_list):
    def gaussian_kernel(distance, h):
        weight = np.exp(-0.5 * ((distance/h)**2))
        return weight

    def predict_one(test_point, training_data, training_returns, h):
        distances = []
        for i in range(len(training_data)):
            dist = np.sqrt(sum((training_data[i][j] - test_point[j])**2 for j in range(len(test_point))))
            distances.append(dist)

        distances = np.array(distances)
        weights = gaussian_kernel(distances, h)
        sum_weights = sum(weights)
        sum_mult = sum(training_returns * weights)
        return sum_mult/sum_weights

    def bandwidth_sel(training_data, training_returns, h_values):
        training_returns = np.array(training_returns)
        training_data = np.array(training_data)

        best_h = None
        best_error = float('inf')

        for h in h_values:
            errors = []
            for i in range(len(training_data)):
                reduced_data = np.delete(training_data, i, axis=0)
                reduced_returns = np.delete(training_returns, i)
                pred = predict_one(training_data[i], reduced_data, reduced_returns, h)
                error = (pred - training_returns[i])**2
                errors.append(error)

            avg_error = np.mean(errors)
            if avg_error < best_error:
                best_error = avg_error
                best_h = h

        return best_h

    cutoff = 600

    scaler = StandardScaler()
    init_train_scaled = scaler.fit_transform(model_df[features_list][:cutoff])
    training_returns = np.array(label[:cutoff])

    h_values = np.arange(1, 30, 1)
    best_h = bandwidth_sel(init_train_scaled, training_returns, h_values)

    mse_scores = []

    while cutoff + 60 <= len(model_df):

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(model_df[features_list][:cutoff])
        test_scaled = scaler.transform(model_df[features_list][cutoff:cutoff+60])

        training_returns = np.array(label[:cutoff])
        y_test = np.array(label[cutoff:cutoff+60])

        predictions = []
        for i in range(len(test_scaled)):
            pred = predict_one(test_scaled[i], train_scaled, training_returns, best_h)
            predictions.append(pred)

        mse_scores.append(mean_squared_error(y_test, predictions))
        cutoff += 60

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(model_df[features_list][:cutoff])
    test_scaled = scaler.transform(model_df[features_list][cutoff:])

    training_returns = np.array(label[:cutoff])
    y_test = np.array(label[cutoff:])

    predictions = []
    for i in range(len(test_scaled)):
        pred = predict_one(test_scaled[i], train_scaled, training_returns, best_h)
        predictions.append(pred)

    mse_scores.append(mean_squared_error(y_test, predictions))

    return mse_scores

def oos_r2(model_mse, baseline_mse):
    return [1-(m/b) for m, b in zip(model_mse, baseline_mse)]

def bootstrap():
    X = model_df[["PE10"]]

    y = label

    reg = LinearRegression()
    reg.fit(X,y)

    coef = reg.coef_[0]

    bootstrap_coefs = []

    for i in range(5000):
        y_shuffled = np.random.permutation(y)
        reg = LinearRegression()
        reg.fit(X,y_shuffled)

        bootstrap_coefs.append(reg.coef_[0])

    bootstrap_coefs = np.array(bootstrap_coefs)

    count = 0

    for cf in bootstrap_coefs:
        if cf <= coef:
            count+=1

    plt.figure(figsize=(9, 5))
    plt.hist(bootstrap_coefs, bins=50, edgecolor='black')
    plt.axvline(coef, color='red', linewidth=2, label=f"Real coefficient ({round(coef, 5)})")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Frequency")
    plt.title("Bootstrap Distribution of PE10 Coefficient Under Null Hypothesis")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bootstrap_dist.png")
    plt.close()

    print("Real coefficient:", coef)
    print("Bootstrap p-value:", count/len(bootstrap_coefs))
    

    

baseline_mse, lr_pe10_mse, times = expanding_window_LR(["PE10"])
_, lr_divy_mse, _ = expanding_window_LR(["Dividend Yield"])
_, lr_combined_mse, _ = expanding_window_LR(["PE10", "Dividend Yield"])
knn_mse = expanding_window_KNN(["PE10", "Dividend Yield"])
kernel_mse = kernel_reg(["PE10", "Dividend Yield"])

lr_pe10_r2 = oos_r2(lr_pe10_mse, baseline_mse)
lr_divy_r2 = oos_r2(lr_divy_mse, baseline_mse)
lr_combined_r2 = oos_r2(lr_combined_mse, baseline_mse)
knn_r2 = oos_r2(knn_mse, baseline_mse)
kernel_r2 = oos_r2(kernel_mse, baseline_mse)

plt.figure(figsize=(9, 5))

plt.plot(times, lr_pe10_r2, label="Linear Regression (PE10)")
plt.plot(times, knn_r2, label="KNN Regression")
plt.plot(times, kernel_r2, label="Kernel Regression")

plt.axhline(y=0, color='black', linestyle='--', linewidth=1, label="Historical Mean Benchmark")

plt.axvline(x="1991", color='red', linestyle=':', linewidth=1)
plt.text("1991", min(lr_pe10_r2) * 0.85, "1991 anomaly", fontsize=7, color='red')

plt.xlabel("Test Period")
plt.ylabel("Out-of-Sample R²")
plt.title("Out-of-Sample R² Over Time by Model")
plt.xticks(fontsize=8)
plt.legend()
plt.tight_layout()
plt.savefig("oos_r2_over_time.png")
plt.close()

"""print("LR PE10 only:", np.mean(lr_pe10_r2))
print("LR Rate only:", np.mean(lr_divy_r2))
print("LR Combined:", np.mean(lr_combined_r2))
print("KNN:", np.mean(knn_r2))
print("Kernel:", np.mean(kernel_r2))

print(f"{'Period':<8} {'LR_PE10':>10} {'LR_DivY':>10} {'LR_Combined':>12} {'KNN':>10} {'Kernel':>10}")
for i, t in enumerate(times):
    print(f"{t:<8} {lr_pe10_r2[i]:>10.3f} {lr_divy_r2[i]:>10.3f} {lr_combined_r2[i]:>12.3f} {knn_r2[i]:>10.3f} {kernel_r2[i]:>10.3f}")

print()
print("Averages:")
print("LR PE10 only:", round(np.mean(lr_pe10_r2), 4))
print("LR DivYield only:", round(np.mean(lr_divy_r2), 4))
print("LR Combined:", round(np.mean(lr_combined_r2), 4))
print("KNN:", round(np.mean(knn_r2), 4))
print("Kernel:", round(np.mean(kernel_r2), 4))


starting_cape = []
cutoff = 600
while cutoff + 60 <= len(features):
    starting_cape.append(round(model_df["PE10"][cutoff], 1))
    cutoff += 60
starting_cape.append(round(model_df["PE10"][cutoff], 1))

print(f"{'Period':<8} {'CAPE':>6} {'LR_PE10':>10} {'LR_DivY':>10} {'LR_Combined':>12} {'KNN':>10} {'Kernel':>10}")
for i, t in enumerate(times):
    print(f"{t:<8} {starting_cape[i]:>6} {lr_pe10_r2[i]:>10.3f} {lr_divy_r2[i]:>10.3f} {lr_combined_r2[i]:>12.3f} {knn_r2[i]:>10.3f} {kernel_r2[i]:>10.3f}")

exclude_year = "1991"
idx = [i for i, t in enumerate(times) if t != exclude_year]

print("\nAverages (all folds):")
print("LR PE10:", round(np.mean(lr_pe10_r2), 4))
print("LR DivYield:", round(np.mean(lr_divy_r2), 4))
print("LR Combined:", round(np.mean(lr_combined_r2), 4))
print("KNN:", round(np.mean(knn_r2), 4))
print("Kernel:", round(np.mean(kernel_r2), 4))

print("\nAverages (excluding 1991):")
print("LR PE10:", round(np.mean([lr_pe10_r2[i] for i in idx]), 4))
print("LR DivYield:", round(np.mean([lr_divy_r2[i] for i in idx]), 4))
print("LR Combined:", round(np.mean([lr_combined_r2[i] for i in idx]), 4))
print("KNN:", round(np.mean([knn_r2[i] for i in idx]), 4))
print("Kernel:", round(np.mean([kernel_r2[i] for i in idx]), 4))
"""

bootstrap()
newey_west_corr()