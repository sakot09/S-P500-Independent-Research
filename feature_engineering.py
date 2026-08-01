import pandas as pd
 
 
def load_data(path="cleaned_sp500.csv"):
    df = pd.read_csv(path)
 
    df["Last Month Price"] = df["SP500"].shift(1)
    df["Growth Factor"] = (df["SP500"] + df["Dividend"] / 12) / df["Last Month Price"]
    df.loc[0, "Growth Factor"] = 1
    df["Cumulative Growth"] = df["Growth Factor"].cumprod()
    df["Real Cumulative Growth"] = (
        df["Cumulative Growth"] * (df["Consumer Price Index"].iloc[-1] / df["Consumer Price Index"])
    )
    df["Future Value"] = df["Real Cumulative Growth"].shift(-120)
    df["Annual Return"] = ((df["Future Value"] / df["Real Cumulative Growth"]) ** (1 / 10)) - 1
    df["Dividend Yield"] = df["Dividend"] / df["SP500"]
 
    model_df = df[df["Annual Return"].notna()].reset_index(drop=True)
    return df, model_df
 
 
df, model_df = load_data()
label = model_df["Annual Return"]