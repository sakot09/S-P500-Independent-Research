import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
 
from feature_engineering import model_df, label
 

def expanding_window_LR(feature):
    cutoff = 600
 
    mse_scores = []
    baseline_mse_scores = []
    times = []
 
    while cutoff + 60 <= len(model_df):
 
        X_train = model_df[feature][:cutoff]
        y_train = label[:cutoff]
 
        X_test = model_df[feature][cutoff:cutoff + 60]
        y_test = label[cutoff:cutoff + 60]
 
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
 
 
def expanding_window_KNN(feature, neighbors):
    
    cutoff = 600
 
    mse_scores = []
    baseline_mse_scores = []
    times = []
 
    while cutoff + 60 <= len(model_df):
 
        X_train = model_df[feature][:cutoff]
        y_train = label[:cutoff]
 
        X_test = model_df[feature][cutoff:cutoff + 60]
        y_test = label[cutoff:cutoff + 60]
 
        knn = KNeighborsRegressor(n_neighbors=neighbors)
 
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
 
    knn = KNeighborsRegressor(n_neighbors=neighbors)
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
 
 
def kernel_reg(features_list):
    
    def gaussian_kernel(distance, h):
        return np.exp(-0.5 * ((distance / h) ** 2))
 
    def predict_one(test_point, training_data, training_returns, h):
        distances = []
        for i in range(len(training_data)):
            dist = np.sqrt(sum((training_data[i][j] - test_point[j]) ** 2 for j in range(len(test_point))))
            distances.append(dist)
        distances = np.array(distances)
        weights = gaussian_kernel(distances, h)
        sum_weights = sum(weights)
        sum_mult = sum(training_returns * weights)
        return sum_mult / sum_weights
 
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
                errors.append((pred - training_returns[i]) ** 2)
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
        test_scaled = scaler.transform(model_df[features_list][cutoff:cutoff + 60])
 
        training_returns = np.array(label[:cutoff])
        y_test = np.array(label[cutoff:cutoff + 60])
 
        predictions = [predict_one(test_scaled[i], train_scaled, training_returns, best_h)
                        for i in range(len(test_scaled))]
 
        mse_scores.append(mean_squared_error(y_test, predictions))
        cutoff += 60
 
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(model_df[features_list][:cutoff])
    test_scaled = scaler.transform(model_df[features_list][cutoff:])
 
    training_returns = np.array(label[:cutoff])
    y_test = np.array(label[cutoff:])
 
    predictions = [predict_one(test_scaled[i], train_scaled, training_returns, best_h)
                    for i in range(len(test_scaled))]
    mse_scores.append(mean_squared_error(y_test, predictions))
 
    return mse_scores, best_h
 
 
def oos_r2(model_mse, baseline_mse):
    return [1 - (m / b) for m, b in zip(model_mse, baseline_mse)]
 
 

def newey_west_corr(feature_list, out=None):
    x = sm.add_constant(model_df[feature_list])
    y = label
 
    z = sm.OLS(y, x)
    naive = z.fit()
    hac = z.fit(cov_type='HAC', cov_kwds={"maxlags": 120})
 
    print(naive.summary(), file=out)
    print(hac.summary(), file=out)
 
    return naive, hac
 
 
def bootstrap(feature=("PE10",), n_iter=5000, plot_name="bootstrap_dist.png"):
    feature = list(feature)
    X = model_df[feature]
    y = label
 
    reg = LinearRegression()
    reg.fit(X, y)
    coef = reg.coef_[0]
 
    bootstrap_coefs = []
    for _ in range(n_iter):
        y_shuffled = np.random.permutation(y)
        reg = LinearRegression()
        reg.fit(X, y_shuffled)
        bootstrap_coefs.append(reg.coef_[0])
 
    bootstrap_coefs = np.array(bootstrap_coefs)
    count = sum(1 for cf in bootstrap_coefs if cf <= coef)
    p_value = count / len(bootstrap_coefs)
 
    plt.figure(figsize=(9, 5))
    plt.hist(bootstrap_coefs, bins=50, edgecolor='black')
    plt.axvline(coef, color='red', linewidth=2, label=f"Real coefficient ({round(coef, 5)})")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Frequency")
    plt.title("Bootstrap Distribution of PE10 Coefficient Under Null Hypothesis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_name)
    plt.close()
 
    return coef, p_value
 
 

def select_k(feature, k_values, plot_name="k_accuracy.png"):
    cutoff = int(0.8 * len(model_df))
 
    X_train = model_df[feature][:cutoff]
    y_train = label[:cutoff]
    X_test = model_df[feature][cutoff:]
    y_test = label[cutoff:]
 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
 
    mse_per_k = []
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        predictions = knn.predict(X_test_scaled)
        mse_per_k.append(mean_squared_error(y_test, predictions))
 
    threshold = 1e-4
    optimal_k = k_values[-1]
    for i in range(1, len(mse_per_k)):
        pct_change = (mse_per_k[i - 1] - mse_per_k[i]) / mse_per_k[i - 1]
        if 0 < pct_change < threshold:
            optimal_k = k_values[i - 1]
            break
 
    plt.figure(figsize=(9, 5))
    plt.plot(k_values, mse_per_k)
    plt.axvline(optimal_k, color='red', linestyle='--', label=f"Selected K = {optimal_k}")
    plt.xlabel("# of Neighbors (K value)")
    plt.ylabel("MSE on held-out 20%")
    plt.title(f"KNN MSE vs K ({', '.join(feature)})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_name)
    plt.close()
 
    return optimal_k, mse_per_k
 
 
def model_accuracy(feature, plot_name="model_accuracy.png"):
    cutoff = int(0.8 * len(model_df))
 
    X_train = model_df[feature][:cutoff]
    y_train = label[:cutoff]
    X_test = model_df[feature][cutoff:]
    y_test = label[cutoff:]
 
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    predictions = linreg.predict(X_test)
 
    plt.figure(figsize=(9, 5))
    plt.scatter(y_test, predictions)
    plt.plot([-.1, .2], [-.1, .2], "--k", label="Correct Prediction")
    plt.xlabel("True Return")
    plt.ylabel("Predicted Return")
    plt.title("Model Accuracy: True vs. Predicted Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_name)
    plt.close()
 
 
def residual_diagnostics(feature, qq_name="qqresidplot.png", resid_name="resid_plot.png"):
    X = model_df[feature]
    y = label
 
    reg = LinearRegression()
    reg.fit(X, y)
    preds = reg.predict(X)
    residuals = y - preds
 
    plt.figure(figsize=(6, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Normal Q-Q Plot of Residuals")
    plt.tight_layout()
    plt.savefig(qq_name)
    plt.close()
 
    plt.figure(figsize=(8, 5))
    plt.scatter(preds, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Annual Return")
    plt.ylabel("Residual")
    plt.title("Residuals vs. Predicted Values")
    plt.tight_layout()
    plt.savefig(resid_name)
    plt.close()
 
 

def starting_cape_by_fold(times):
    starting_cape = []
    cutoff = 600
    for _ in range(len(times) - 1):
        starting_cape.append(round(model_df["PE10"][cutoff], 1))
        cutoff += 60
    starting_cape.append(round(model_df["PE10"][cutoff], 1))
    return starting_cape
 
 
def exclude_fold(values, times, exclude="1991"):
    idx = [i for i, t in enumerate(times) if t != exclude]
    return [values[i] for i in idx]
 
 

if __name__ == "__main__":
 
    baseline_mse, lr_pe10_mse, times = expanding_window_LR(["PE10"])
    knn_mse_single = expanding_window_KNN(["PE10"], neighbors=105)
    kernel_mse_single, h_single = kernel_reg(["PE10"])
 
    lr_pe10_r2 = oos_r2(lr_pe10_mse, baseline_mse)
    knn_r2_single = oos_r2(knn_mse_single, baseline_mse)
    kernel_r2_single = oos_r2(kernel_mse_single, baseline_mse)
 
    optimal_k_single, mse_per_k_single = select_k(["PE10"], list(range(1, 601, 5)),
                                                    plot_name="k_accuracy_single.png")
    model_accuracy(["PE10"], plot_name="model_accuracy_single.png")
    residual_diagnostics(["PE10"], qq_name="qqresidplot_single.png", resid_name="resid_plot_single.png")
 
    coef, bootstrap_p = bootstrap(feature=("PE10",))
 
    starting_cape = starting_cape_by_fold(times)
 
    plt.figure(figsize=(9, 5))
    plt.plot(times, lr_pe10_r2, label="Linear Regression (PE10)")
    plt.plot(times, knn_r2_single, label="KNN Regression (PE10, K=105)")
    plt.plot(times, kernel_r2_single, label=f"Kernel Regression (PE10, h={h_single})")
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, label="Historical Mean Benchmark")
    plt.axvline(x="1991", color='red', linestyle=':', linewidth=1)
    plt.text("1991", min(lr_pe10_r2) * 0.85, "1991 anomaly", fontsize=7, color='red')
    plt.xlabel("Test Period")
    plt.ylabel("Out-of-Sample R²")
    plt.title("Out-of-Sample R² Over Time — Single Predictor (PE10)")
    plt.xticks(fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig("oos_r2_over_time_single.png")
    plt.close()
 
    with open("results_expanding_single.txt", "w") as f:
        print("EXPANDING WINDOW — SINGLE PREDICTOR (PE10 only)\n", file=f)
 
        print(f"Optimal K (from MSE-vs-K curve): {optimal_k_single}  [fixed at 105 for all reported runs]", file=f)
        print(f"Optimal bandwidth h: {h_single}\n", file=f)
 
        print("Average OOS-R2 (all folds):", file=f)
        print(f"  LR:     {np.mean(lr_pe10_r2):.4f}", file=f)
        print(f"  KNN:    {np.mean(knn_r2_single):.4f}", file=f)
        print(f"  Kernel: {np.mean(kernel_r2_single):.4f}\n", file=f)
 
        lr_excl = exclude_fold(lr_pe10_r2, times)
        knn_excl = exclude_fold(knn_r2_single, times)
        kernel_excl = exclude_fold(kernel_r2_single, times)
 
        print("Average OOS-R2 (excluding 1991):", file=f)
        print(f"  LR:     {np.mean(lr_excl):.4f}", file=f)
        print(f"  KNN:    {np.mean(knn_excl):.4f}", file=f)
        print(f"  Kernel: {np.mean(kernel_excl):.4f}\n", file=f)
 
        print("Average MSE:", file=f)
        print(f"  Historical Mean baseline: {np.mean(baseline_mse):.6f}", file=f)
        print(f"  LR:     {np.mean(lr_pe10_mse):.6f}", file=f)
        print(f"  KNN:    {np.mean(knn_mse_single):.6f}", file=f)
        print(f"  Kernel: {np.mean(kernel_mse_single):.6f}\n", file=f)
 
        print("Fold-by-fold OOS-R2 with starting CAPE:", file=f)
        print(f"{'Period':<8} {'CAPE':>6} {'LR':>8} {'KNN':>8} {'Kernel':>8}", file=f)
        for i, t in enumerate(times):
            print(f"{t:<8} {starting_cape[i]:>6} {lr_pe10_r2[i]:>8.3f} {knn_r2_single[i]:>8.3f} {kernel_r2_single[i]:>8.3f}", file=f)
 
        print("\nBootstrap permutation test (5,000 iterations):", file=f)
        print(f"  Real PE10 coefficient: {coef:.6f}", file=f)
        print(f"  Bootstrap p-value: {bootstrap_p}\n", file=f)
 
        print("Newey-West / naive OLS (PE10 only):\n", file=f)
        newey_west_corr(["PE10"], out=f)
 
    _, lr_divy_mse, _ = expanding_window_LR(["Dividend Yield"])
    _, lr_combined_mse, _ = expanding_window_LR(["PE10", "Dividend Yield"])
    knn_mse_multi = expanding_window_KNN(["PE10", "Dividend Yield"], neighbors=223)
    kernel_mse_multi, h_multi = kernel_reg(["PE10", "Dividend Yield"])
 
    lr_divy_r2 = oos_r2(lr_divy_mse, baseline_mse)
    lr_combined_r2 = oos_r2(lr_combined_mse, baseline_mse)
    knn_r2_multi = oos_r2(knn_mse_multi, baseline_mse)
    kernel_r2_multi = oos_r2(kernel_mse_multi, baseline_mse)
 
    optimal_k_multi, mse_per_k_multi = select_k(["PE10", "Dividend Yield"], list(range(1, 601, 5)),
                                                  plot_name="k_accuracy_multi.png")
 
    plt.figure(figsize=(9, 5))
    plt.plot(times, lr_pe10_r2, label="LR (PE10 only)")
    plt.plot(times, lr_divy_r2, label="LR (Dividend Yield only)")
    plt.plot(times, lr_combined_r2, label="LR (PE10 + Dividend Yield)")
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, label="Historical Mean Benchmark")
    plt.xlabel("Test Period")
    plt.ylabel("Out-of-Sample R²")
    plt.title("Does Adding Dividend Yield Improve Linear Regression?")
    plt.xticks(fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig("oos_r2_dividend_yield_test.png")
    plt.close()
 
    plt.figure(figsize=(9, 5))
    plt.plot(times, lr_combined_r2, label="Linear Regression (PE10 + DivYield)")
    plt.plot(times, knn_r2_multi, label="KNN Regression (PE10 + DivYield, K=223)")
    plt.plot(times, kernel_r2_multi, label=f"Kernel Regression (PE10 + DivYield, h={h_multi})")
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, label="Historical Mean Benchmark")
    plt.xlabel("Test Period")
    plt.ylabel("Out-of-Sample R²")
    plt.title("Out-of-Sample R² Over Time — Multi Predictor (PE10 + Dividend Yield)")
    plt.xticks(fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig("oos_r2_over_time_multi.png")
    plt.close()
 
    with open("results_expanding_multi.txt", "w") as f:
        print("EXPANDING WINDOW — MULTI PREDICTOR (PE10 + Dividend Yield)\n", file=f)
 
        print(f"Optimal K (from MSE-vs-K curve): {optimal_k_multi}  [fixed at 223 for all reported runs]", file=f)
        print(f"Optimal bandwidth h: {h_multi}\n", file=f)
 
        print("Average OOS-R2 (all folds):", file=f)
        print(f"  LR (PE10 only):     {np.mean(lr_pe10_r2):.4f}", file=f)
        print(f"  LR (DivYield only): {np.mean(lr_divy_r2):.4f}", file=f)
        print(f"  LR (Combined):      {np.mean(lr_combined_r2):.4f}", file=f)
        print(f"  KNN (Combined):     {np.mean(knn_r2_multi):.4f}", file=f)
        print(f"  Kernel (Combined):  {np.mean(kernel_r2_multi):.4f}\n", file=f)
 
        idx_excl = [i for i, t in enumerate(times) if t != "1991"]
        print("Average OOS-R2 (excluding 1991):", file=f)
        print(f"  LR (PE10 only):     {np.mean([lr_pe10_r2[i] for i in idx_excl]):.4f}", file=f)
        print(f"  LR (DivYield only): {np.mean([lr_divy_r2[i] for i in idx_excl]):.4f}", file=f)
        print(f"  LR (Combined):      {np.mean([lr_combined_r2[i] for i in idx_excl]):.4f}", file=f)
        print(f"  KNN (Combined):     {np.mean([knn_r2_multi[i] for i in idx_excl]):.4f}", file=f)
        print(f"  Kernel (Combined):  {np.mean([kernel_r2_multi[i] for i in idx_excl]):.4f}\n", file=f)
 
        print("Fold-by-fold OOS-R2 with starting CAPE:", file=f)
        print(f"{'Period':<8} {'CAPE':>6} {'LR_PE10':>9} {'LR_DivY':>9} {'LR_Comb':>9} {'KNN':>8} {'Kernel':>8}", file=f)
        for i, t in enumerate(times):
            print(f"{t:<8} {starting_cape[i]:>6} {lr_pe10_r2[i]:>9.3f} {lr_divy_r2[i]:>9.3f} "
                  f"{lr_combined_r2[i]:>9.3f} {knn_r2_multi[i]:>8.3f} {kernel_r2_multi[i]:>8.3f}", file=f)
 
        print("\nNewey-West / naive OLS (PE10 + Dividend Yield):\n", file=f)
        newey_west_corr(["PE10", "Dividend Yield"], out=f)
 
