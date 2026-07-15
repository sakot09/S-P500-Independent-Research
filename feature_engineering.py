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

cutoff = int(.8*len(model_df))

X_train = features[:cutoff]
y_train = label[:cutoff]

X_test = features[cutoff:]
y_test = label[cutoff:]

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

predictions = lin_reg.predict(X_test)

plt.figure(figsize=(8, 8))


plt.scatter(y_test, predictions)
plt.plot([0, 1], [0, 1], '--k', label="Correct prediction")


plt.xlabel('True Return')
plt.ylabel('Predicted Return')
plt.title("Real vs. Predicted Return")

plt.axis('tight')
plt.legend()
plt.tight_layout()
plt.savefig("model_accuracy.png")


