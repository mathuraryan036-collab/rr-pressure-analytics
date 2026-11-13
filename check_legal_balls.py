import os
import pandas as pd

DATA_DIR = r"C:\Users\ASUS\.gemini\antigravity\scratch\RR"
bbb = pd.read_csv(os.path.join(DATA_DIR, "ball_by_ball.csv"))

# Check match 13511 inn 1
m1 = bbb[(bbb['match_id'] == 13511) & (bbb['innings'] == 1)]
is_legal = ~m1['extras_type'].isin(['wide', 'Overstep'])
print("Total rows:", len(m1))
print("Legal balls count:", is_legal.sum())
print("Overid min-max:", m1['overid'].min(), m1['overid'].max())
print("Extras types in m1:\n", m1['extras_type'].value_counts(dropna=False))
