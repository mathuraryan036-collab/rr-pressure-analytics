# Rajasthan Royals: SupeRR Selector (Season 2)
## Round 1: Identifying the Most Reliable Players Under Pressure

An end-to-end, reproducible sports analytics system built for the **Rajasthan Royals SupeRR Selector Season 2 (Round 1)**.

This repository analyzes delivery-level records across **200,999 balls from 867 IPL and T20 International (T20I) matches (2022–2026)** to identify and rank the **Top 10 Batters** and **Top 10 Bowlers** who deliver most reliably in high-pressure match situations compared with their usual baselines.

---

## 🏆 Final Submission Rankings

### Top 10 Batters Under Pressure
| Rank | Player Name | Pressure Balls | Pressure SR | Baseline SR | Strike Rate Delta ($\Delta\text{SR}$) | True SR Diff vs Par | Match Consistency | Pressure Rating (0–100) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Naman Dhir** | 233 | **200.43** | 168.42 | **+32.01** | **+31.99** | **69.6%** | **100.0** |
| **2** | **Jos Buttler** | 357 | **195.80** | 155.39 | **+40.41** | **+20.42** | **71.4%** | **87.9** |
| **3** | **Fakhar Zaman** | 129 | **185.27** | 131.40 | **+53.87** | **+21.18** | **76.9%** | **82.9** |
| **4** | **Babar Azam** | 110 | **198.18** | 133.19 | **+64.99** | **+17.06** | **60.0%** | **80.6** |
| **5** | **Rishabh Pant** | 362 | **185.08** | 143.06 | **+42.02** | **+17.08** | **56.3%** | **73.4** |
| **6** | **Najibullah Zadran** | 136 | **176.47** | 138.56 | **+37.91** | **+11.63** | **61.5%** | **73.2** |
| **7** | **Suryakumar Yadav** | 623 | **195.51** | 166.43 | **+29.08** | **+18.46** | **66.7%** | **72.2** |
| **8** | **Moeen Ali** | 314 | **181.21** | 142.12 | **+39.09** | **+22.51** | **71.0%** | **70.4** |
| **9** | **Shubman Gill** | 235 | **191.49** | 149.39 | **+42.10** | **+14.74** | **60.9%** | **63.0** |
| **10** | **Sikandar Raza** | 319 | **172.73** | 135.53 | **+37.20** | **+20.75** | **66.7%** | **61.8** |

### Top 10 Bowlers Under Pressure
| Rank | Player Name | Pressure Balls | Pressure ER | Baseline ER | True RPO Saved vs Par | Wicket % Under Pressure | Dot Ball % Under Pressure | Match Consistency | Pressure Rating (0–100) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Craig Young** | 117 | **7.59** | 8.45 | **+2.06** | **11.1%** | 39.3% | **100.0%** | **100.0** |
| **2** | **Sufiyan Muqeem** | 183 | **5.21** | 5.92 | **+1.23** | **8.7%** | **57.9%** | **92.9%** | **92.8** |
| **3** | **Umran Malik** | 150 | **8.44** | 9.14 | **+1.74** | **11.3%** | 42.0% | **78.9%** | **89.9** |
| **4** | **Sunil Narine** | 424 | **6.67** | 6.76 | **+2.62** | **6.4%** | 37.5% | **85.7%** | **85.2** |
| **5** | **Ehsan Khan** | 131 | **7.83** | 6.87 | **+2.60** | **8.4%** | 37.4% | **100.0%** | **84.6** |
| **6** | **Matheesha Pathirana** | 591 | **7.82** | 7.80 | **+2.02** | **8.8%** | 37.4% | **85.7%** | **84.4** |
| **7** | **Spencer Johnson** | 122 | **8.70** | 8.84 | **+1.72** | **8.2%** | 40.2% | **100.0%** | **83.1** |
| **8** | **Jamie Overton** | 216 | **7.78** | 8.27 | **+1.06** | **10.2%** | **44.0%** | **89.5%** | **81.7** |
| **9** | **Roston Chase** | 100 | **6.72** | 7.59 | **+3.32** | **6.0%** | 40.0% | **93.3%** | **81.3** |
| **10** | **Jasprit Bumrah** | 646 | **7.05** | 6.85 | **+2.51** | **7.3%** | 37.0% | **82.3%** | **81.1** |

---

## 🔬 Mathematical Architecture

### 1. Ball-by-Ball Continuous Pressure Index ($\text{PI}$)
* **2nd Innings (Chases)**:
  $$\text{Strain}_{\text{RRR}, t} = \min\left(2.5, \max\left(0.3, \frac{\text{RRR}_t}{8.5}\right)\right)$$
  $$\text{Stress}_{\text{Wkt}, t} = \min\left(2.0, \max\left(0.4, \frac{11 - W_{\text{rem}, t}}{5.0}\right)\right)$$
  $$\text{Urgency}_t = 0.5 + 0.5 \times \left(\frac{120 - B_{\text{rem}, t}}{120}\right)$$
* **1st Innings (Setting Target)**: Crisis weights for top-order collapses (Powerplay wickets $\ge 2$) and death over maximization (Overs 18–20 carry $1.50\times$).
* **Stakes Multiplier**: $1.20\times$ for knockout and playoff fixtures.
* **Top 30% Isolation**: Deliveries with $\text{PI} \ge 0.70$ are classified as **High Pressure**.

### 2. Bayesian Shrinkage & Credibility
Low-sample variance is regressed toward the population mean using Empirical Bayes shrinkage ($K = 30$ deliveries):
$$\hat{\theta}_i = \frac{N_i}{N_i + K} \theta_i + \frac{K}{N_i + K} \bar{\theta}_{\text{pop}}$$

### 3. Contextual Par Adjustment & Competition Tiers
Expected strike rate and economy rates are calibrated per match context. Competition tiers apply $1.25\times$ for IPL and $1.00\times$ for T20I Full Members.

---

## 📊 Visual Assets

* `charts/pressure_index_distribution.png`: Continuous delivery pressure distribution.
* `charts/batting_clutch_scatter.png`: Quadrant chart mapping Baseline SR vs High-Pressure SR.
* `charts/bowling_clutch_scatter.png`: Run suppression vs 9.03 global par ER.
* `charts/consistency_comparison.png`: Repeat match consistency rates.
* `charts/journey_showcase_poster.png`: 4-in-1 executive showcase poster.

---

## 🚀 How to Reproduce

1. **Clone this repository**:
   ```bash
   git clone https://github.com/mathuraryan036-collab/rr-pressure-analytics.git
   cd rr-pressure-analytics
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the pipeline**:
   ```bash
   python run_pipeline.py
   ```
   *(Regenerates all rankings, CSV exports, and performance metrics).*
