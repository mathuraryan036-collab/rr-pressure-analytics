import os
import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\ASUS\.gemini\antigravity\scratch\RR"
bbb = pd.read_csv(os.path.join(DATA_DIR, "ball_by_ball.csv"))
# Let's inspect some actual death over chases in inn 2
