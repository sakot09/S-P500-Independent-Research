import pandas as pd


df = pd.read_csv("sp500.csv")

df["Date"] = pd.to_datetime(df["Date"])


df.loc[df["PE10"] == 0, "PE10"] = pd.NA

for col in df.columns:
    df.loc[df["PE10"] == 0, "PE10"] = pd.NA


df.dropna(axis=0, how="any", subset=["PE10"], inplace= True)
df.to_csv("cleaned_sp500.csv", index=False)