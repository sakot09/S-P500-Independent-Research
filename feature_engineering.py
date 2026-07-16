import pandas as pd
from sklearn import  model_selection
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_sp500.csv")

df["Last Month Price"] = df["SP500"].shift(1, axis=0)
df["Growth Factor"] = (df["SP500"]/df["Last Month Price"]) + (df['Dividend']/df['Last Month Price'])

df.loc[0,"Growth Factor"] = 1
df["Cumulative Growth"] = df["Growth Factor"].cumprod()
df["Real Cumulative Growth"] = df["Cumulative Growth"] * (df["Consumer Price Index"].iloc[-1] / df['Consumer Price Index'])

df["Future Value"] = df["Real Cumulative Growth"].shift(-120)
df["Annual Return"] = ((df["Future Value"]/df["Real Cumulative Growth"])**(1/10))-1

model_df = df[df["Annual Return"].notna()].reset_index(drop=True)

features = model_df[["PE10"]]
label = model_df["Annual Return"]




cutoff = .8*len(features)

mse_scores = []
times = []

residuals = []

baseline_mse_scores = []

predicted_values = []

X_train = features
y_train = label

X_test = features
y_test = label

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

predictions = lin_reg.predict(X_test)

predicted_values.extend(predictions)

resid = y_test - predictions

residuals.extend(resid)

plt.figure(figsize=(8,5))

plt.scatter(predicted_values, residuals)

plt.axhline(y=0, color='red', linestyle='--')

plt.xlabel("Predicted Annual Return")
plt.ylabel("Residual")
plt.title("Residuals vs. Predicted Values")

plt.show()

"""while cutoff + 60 <= len(features):


    X_train = features[:cutoff]
    y_train = label[:cutoff]

    X_test = features[cutoff:cutoff+60]
    y_test = label[cutoff:cutoff+60]

    baseline_prediction = [y_train.mean()] * len(y_test)

    baseline_mse = mean_squared_error(y_test, baseline_prediction)

    baseline_mse_scores.append(baseline_mse)    

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    predictions = lin_reg.predict(X_test)

    
    mse = mean_squared_error(y_test, predictions)

    mse_scores.append(mse)
    
    time_period = model_df['Date'][cutoff]

    times.append(time_period)

    cutoff+=60

X_train = features[:cutoff]
y_train = label[:cutoff]

X_test = features[cutoff:len(features)]
y_test = label[cutoff:len(features)]

baseline_prediction = [y_train.mean()] * len(y_test)

baseline_mse = mean_squared_error(y_test, baseline_prediction)

baseline_mse_scores.append(baseline_mse)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

predictions = lin_reg.predict(X_test)



mse = mean_squared_error(y_test, predictions)
mse_scores.append(mse)
    
time_period = model_df['Date'][cutoff]
times.append(time_period)

for i in range(len(times)):
    year = times[i][:4]
    times[i] = year"""






