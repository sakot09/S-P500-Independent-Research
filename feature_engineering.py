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




cutoff = 600

mse_scores = []
times = []

while cutoff + 60 <= len(features):


    X_train = features[:cutoff]
    y_train = label[:cutoff]

    X_test = features[cutoff:cutoff+60]
    y_test = label[cutoff:cutoff+60]

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

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

predictions = lin_reg.predict(X_test)

mse = mean_squared_error(y_test, predictions)
mse_scores.append(mse)
    
time_period = model_df['Date'][cutoff]
times.append(time_period)

for i in range(len(times)):
    year = times[i][:4]
    times[i] = year

plt.plot(times, mse_scores)

plt.xlabel("Date")
plt.xticks(fontsize = 7)
plt.ylabel("MSE")
plt.title("MSE of Model Over Time")

plt.savefig("mse_ot.png")




