import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from feature_engineering import model_df, label

from expanding_window_engineering import (
    expanding_window_LR,
    expanding_window_KNN,
    kernel_reg,
    oos_r2,
    starting_cape_by_fold,
)

WINDOW_SIZE = 600  

def rolling_window_LR(feature):
    cutoff = WINDOW_SIZE
    beginning = 0

    mse_scores = []
    baseline_mse_scores = []
    times = []

    while cutoff + 60 <= len(model_df):
        X_train = model_df[feature][beginning:cutoff]
        y_train = label[beginning:cutoff]
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
        beginning += 60

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


def rolling_window_KNN(feature, neighbors):
    cutoff = WINDOW_SIZE
    beginning = 0

    mse_scores = []
    times = []

    while cutoff + 60 <= len(model_df):
        X_train = model_df[feature][beginning:cutoff]
        y_train = label[beginning:cutoff]
        X_test = model_df[feature][cutoff:cutoff + 60]
        y_test = label[cutoff:cutoff + 60]

        knn = KNeighborsRegressor(n_neighbors=neighbors)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        knn.fit(X_train_scaled, y_train)
        predictions = knn.predict(X_test_scaled)
        mse_scores.append(mean_squared_error(y_test, predictions))

        times.append(model_df["Date"][cutoff][:4])
        cutoff += 60
        beginning += 60

    X_train = model_df[feature][beginning:cutoff]
    y_train = label[beginning:cutoff]
    X_test = model_df[feature][cutoff:]
    y_test = label[cutoff:]

    knn = KNeighborsRegressor(n_neighbors=neighbors)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn.fit(X_train_scaled, y_train)
    predictions = knn.predict(X_test_scaled)
    mse_scores.append(mean_squared_error(y_test, predictions))
    times.append(model_df["Date"][cutoff][:4])

    return mse_scores


def rolling_kernel_reg(features_list):
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

    cutoff = WINDOW_SIZE
    beginning = 0

    scaler = StandardScaler()
    init_train_scaled = scaler.fit_transform(model_df[features_list][beginning:cutoff])
    training_returns = np.array(label[beginning:cutoff])

    h_values = np.arange(1, 30, 1)
    best_h = bandwidth_sel(init_train_scaled, training_returns, h_values)

    mse_scores = []

    while cutoff + 60 <= len(model_df):
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(model_df[features_list][beginning:cutoff])
        test_scaled = scaler.transform(model_df[features_list][cutoff:cutoff + 60])

        training_returns = np.array(label[beginning:cutoff])
        y_test = np.array(label[cutoff:cutoff + 60])

        predictions = [predict_one(test_scaled[i], train_scaled, training_returns, best_h)
                        for i in range(len(test_scaled))]

        mse_scores.append(mean_squared_error(y_test, predictions))
        cutoff += 60
        beginning += 60

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(model_df[features_list][beginning:cutoff])
    test_scaled = scaler.transform(model_df[features_list][cutoff:])

    training_returns = np.array(label[beginning:cutoff])
    y_test = np.array(label[cutoff:])

    predictions = [predict_one(test_scaled[i], train_scaled, training_returns, best_h)
                    for i in range(len(test_scaled))]
    mse_scores.append(mean_squared_error(y_test, predictions))

    return mse_scores, best_h


if __name__ == "__main__":

    rol_base_mse, rol_lr_mse, times = rolling_window_LR(["PE10"])
    rol_knn_mse = rolling_window_KNN(["PE10"], neighbors=105)
    rol_kernel_mse, rol_h = rolling_kernel_reg(["PE10"])

    exp_base_mse, exp_lr_mse, exp_times = expanding_window_LR(["PE10"])
    exp_knn_mse = expanding_window_KNN(["PE10"], neighbors=105)
    exp_kernel_mse, exp_h = kernel_reg(["PE10"])

    lr_rolling = oos_r2(rol_lr_mse, rol_base_mse)
    lr_expanding = oos_r2(exp_lr_mse, exp_base_mse)

    knn_rolling = oos_r2(rol_knn_mse, rol_base_mse)
    knn_expanding = oos_r2(exp_knn_mse, exp_base_mse)

    kernel_rolling = oos_r2(rol_kernel_mse, rol_base_mse)
    kernel_expanding = oos_r2(exp_kernel_mse, exp_base_mse)

    starting_cape = starting_cape_by_fold(times)

    def comparison_plot(rolling_vals, expanding_vals, model_name, filename):
        plt.figure(figsize=(10, 5))
        plt.plot(times, rolling_vals, label='Rolling OOS-R^2', color='blue', lw=2)
        plt.plot(times, expanding_vals, label='Expanding OOS-R^2', color='orange', lw=2)
        plt.axhline(0, color='red', linestyle=':', label='Benchmark (y=0)')
        plt.title(f'Out-of-Sample R^2 Comparison — {model_name} (PE10 only)')
        plt.xlabel('Date')
        plt.ylabel('OOS-R^2')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    comparison_plot(lr_rolling, lr_expanding, "Linear Regression", "window_validation_comparison_lr.png")
    comparison_plot(knn_rolling, knn_expanding, "KNN Regression (K=105)", "window_validation_comparison_knn.png")
    comparison_plot(kernel_rolling, kernel_expanding, f"Kernel Regression (h={rol_h})", "window_validation_comparison_kernel.png")

    plt.figure(figsize=(10, 5))
    plt.plot(times, lr_rolling, label='Rolling LR OOS-R^2', color='black', lw=2)
    plt.plot(times, knn_rolling, label='Rolling KNN OOS-R^2', color='orange', lw=2)
    plt.plot(times, kernel_rolling, label='Rolling Kernel OOS-R^2', color='blue', lw=2)
    plt.axhline(0, color='red', linestyle=':', label='Benchmark (y=0)')
    plt.title('Out-of-Sample R^2 Comparison — Rolling Window, All Models (PE10 only)')
    plt.xlabel('Date')
    plt.ylabel('OOS-R^2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("oos_r2_ot_rolling.png")
    plt.close()

    with open("results_rolling.txt", "w") as f:
        print("ROLLING WINDOW RESULTS — SINGLE PREDICTOR (PE10 only)\n", file=f)
        print(f"Window size: {WINDOW_SIZE} rows (~50 years)", file=f)
        print(f"Rolling KNN neighbors: 105 (matches expanding single-pred K)", file=f)
        print(f"Rolling kernel bandwidth h: {rol_h} (expanding kernel h: {exp_h})\n", file=f)

        print("Average OOS-R2, Rolling (all folds):", file=f)
        print(f"  LR:     {np.mean(lr_rolling):.4f}", file=f)
        print(f"  KNN:    {np.mean(knn_rolling):.4f}", file=f)
        print(f"  Kernel: {np.mean(kernel_rolling):.4f}\n", file=f)

        idx = [i for i, t in enumerate(times) if t != "1991"]
        print("Average OOS-R2, Rolling (excluding 1991):", file=f)
        print(f"  LR:     {np.mean([lr_rolling[i] for i in idx]):.4f}", file=f)
        print(f"  KNN:    {np.mean([knn_rolling[i] for i in idx]):.4f}", file=f)
        print(f"  Kernel: {np.mean([kernel_rolling[i] for i in idx]):.4f}\n", file=f)

        print("Expanding vs. Rolling, average OOS-R2 (excluding 1991):", file=f)
        print(f"  LR      — Expanding: {np.mean([lr_expanding[i] for i in idx]):.4f}   "
              f"Rolling: {np.mean([lr_rolling[i] for i in idx]):.4f}", file=f)
        print(f"  KNN     — Expanding: {np.mean([knn_expanding[i] for i in idx]):.4f}   "
              f"Rolling: {np.mean([knn_rolling[i] for i in idx]):.4f}", file=f)
        print(f"  Kernel  — Expanding: {np.mean([kernel_expanding[i] for i in idx]):.4f}   "
              f"Rolling: {np.mean([kernel_rolling[i] for i in idx]):.4f}\n", file=f)

        print("Fold-by-fold OOS-R2 (Rolling vs Expanding, all three models), with starting CAPE:", file=f)
        print(f"{'Period':<8} {'CAPE':>6} {'LR_Roll':>8} {'LR_Exp':>8} {'KNN_Roll':>9} {'KNN_Exp':>8} "
              f"{'Ker_Roll':>9} {'Ker_Exp':>8}", file=f)
        for i, t in enumerate(times):
            print(f"{t:<8} {starting_cape[i]:>6} {lr_rolling[i]:>8.3f} {lr_expanding[i]:>8.3f} "
                  f"{knn_rolling[i]:>9.3f} {knn_expanding[i]:>8.3f} "
                  f"{kernel_rolling[i]:>9.3f} {kernel_expanding[i]:>8.3f}", file=f)

