import matplotlib.pyplot as plt
import statsmodels.api as sm
 
from feature_engineering import model_df, label
 
f_stats = []
p_values = []
breaks = []
dates = []
results_p = []
results_f = []
 
for i in range(238, 1353):
 
    dummy = (model_df.index >= i).astype(int)
    interaction = dummy * model_df["PE10"]
 
    X = model_df[["PE10"]].copy()
    X["dummy"] = dummy
    X["interaction"] = interaction
    X = sm.add_constant(X)
 
    z = sm.OLS(label, X)
    result = z.fit(cov_type='HAC', cov_kwds={"maxlags": 120})
 
    f_test = result.f_test("dummy = 0, interaction = 0")
    f_stats.append(f_test.fvalue)
    p_values.append(float(f_test.pvalue))
    dates.append(model_df["Date"][i])
 
bonferroni_threshold = 0.05 / 1115
 
for i in range(len(p_values)):
    if p_values[i] < bonferroni_threshold:
        results_p.append(p_values[i])
        results_f.append(f_stats[i])
        breaks.append(dates[i])
 
with open("research_results.txt", "w") as f:
    print("Chow Test Results\n", file=f)
    print(f"Bonferroni threshold: {bonferroni_threshold}", file=f)
    print(f"Number of significant break points: {len(breaks)}", file=f)
    if breaks:
        print(f"Break window: {breaks[0]} to {breaks[-1]}", file=f)
        peak_idx = results_f.index(max(results_f))
        print(f"Peak F-statistic: {results_f[peak_idx]:.4f} at {breaks[peak_idx]}", file=f)
    print(f"\nDates of Breaks: {breaks}", file=f)
    print(f"F-Statistics of Breaks: {results_f}", file=f)
    print(f"P-Values of Breaks: {results_p}", file=f)
 
critical_f = min(
    (f_stats[i] for i in range(len(p_values))
     if abs(p_values[i] - bonferroni_threshold) == min(abs(p - bonferroni_threshold) for p in p_values)),
    default=None
)
 
plt.figure(figsize=(12, 5))
plt.plot(dates, f_stats, color='blue', linewidth=1)
plt.axhline(y=critical_f, color='red', linestyle='--',
            label=f'Bonferroni threshold (p={round(bonferroni_threshold, 6)})')
 
if breaks:
    plt.axvspan(breaks[0], breaks[-1], color='orange', alpha=0.15,
                label=f'Significant break window ({breaks[0][:4]}\u2013{breaks[-1][:4]})')
 
plt.xlabel("Date")
plt.ylabel("F-Statistic")
plt.title("Sequential Chow Test — F-Statistic Over Time")
 
tick_spacing = 120
tick_indices = list(range(0, len(dates), tick_spacing))
plt.xticks(ticks=tick_indices, labels=[dates[i][:4] for i in tick_indices], rotation=0, fontsize=8)
 
plt.legend()
plt.tight_layout()
plt.savefig("figures/structural_break.png")
plt.close()
 
