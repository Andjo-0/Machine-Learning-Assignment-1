import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif
from scipy.stats import f_oneway

def correlation_ratio(categories, values):
    """
    Correlation ratio (eta squared) for categorical -> numerical.
    """
    categories = pd.Series(categories)
    values = pd.Series(values)
    cat_groups = [values[categories == cat] for cat in categories.unique()]
    n_total = len(values)
    grand_mean = values.mean()
    ss_between = sum([len(g) * (g.mean() - grand_mean) ** 2 for g in cat_groups])
    ss_total = ((values - grand_mean) ** 2).sum()
    return np.sqrt(ss_between / ss_total)

datapath = "Data/AmesHousing.csv"

df = pd.read_csv(datapath)

target = "SalePrice"   # <-- your target column name
cat_cols = df.select_dtypes(exclude='number').columns

corr_scores = {}
for col in cat_cols:
    eta = correlation_ratio(df[col], df[target])
    corr_scores[col] = eta

# Sort and pick top 6
top6_cats = sorted(corr_scores, key=corr_scores.get, reverse=True)[:6]
print(top6_cats)


import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 2 rows × 3 columns
axes = axes.flatten()

for ax, col in zip(axes, top6_cats):
    sns.boxplot(x=col, y=target, data=df, ax=ax)
    ax.set_title(f"{col} vs {target}", fontsize=12)
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.show()
