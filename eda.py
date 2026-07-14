import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




df = pd.read_csv("cleaned_sp500.csv")

df["Date"] = pd.to_datetime(df["Date"])

print(df.info())
print(df.describe())

df.hist(figsize=(15,10))
plt.tight_layout()
plt.savefig("vardist.png")


for col in df.columns[1:]:
    plt.figure(figsize=(12,5))
    plt.plot(df["Date"], df[col])
    plt.title("Variables Over Time")
    plt.xlabel("Date")
    plt.ylabel(f"{col}")
    plt.savefig(f"{col} Graph.png")

corr = df.corr(numeric_only=True)

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.savefig("heatmap.png")