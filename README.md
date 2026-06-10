# FIFA World Cup 2026 Predictor

EV-optimised prediction model for the sport5 "5 חבר'ה" group-stage game.  
Goal: maximise expected sport5 points, not just predict the most likely result.

---

## How the sport5 scoring works

- Correct result (W/D/L) → base odds points
- Exact scoreline → base odds **+ 4** bonus points (group stage) / **+ 6** (knockout)

So the EV of predicting score G1-G2 is:

```
EV = odds[result] × P(result) + 4 × P(exact G1-G2)
```

This means picking a high-odds underdog can beat picking the favourite even if the favourite is more likely to win.

---

## Architecture (4 files)

| File | Role |
|------|------|
| `data.py` | 48 teams, 12 groups, 72 fixtures, squad strength, `SPORT5_ODDS` |
| `models.py` | ELO system, Dixon-Coles, XGBoost ML, Ensemble |
| `predict.py` | EV computation, strategy printer, pie charts, heat-maps |
| `main.py` | Orchestration — runs everything end-to-end |

---

## Models

### 1. Dynamic ELO
- Initialised at 1500 for all teams, updated from 6 years of real match history
- K parameter (learning rate) **tuned automatically** via Brier-score minimisation on a held-out window — converged to K ≈ 41–52 depending on data period
- Tournament weights: WC final × 1.5, friendlies × 0.5, qualifiers × 1.0
- Win vs stronger opponent → larger rating gain (standard ELO expected-score formula)
- Updated in-tournament after every WC match (rounds 2 & 3 benefit from round 1 ELO shift)

### 2. Dixon-Coles (DC)
- Bivariate Poisson with ρ-correction for low-score correlations (0-0, 1-0, 0-1, 1-1)
- Attack/defence parameters fitted via vectorised NLL optimisation on real data
- λ values scaled by ELO ratio at prediction time (dynamic ELO weight grows with ELO gap)
- Home advantage only for Mexico, Canada, USA (tournament hosts)

### 3. ML (XGBoost + Poisson regressors)
- XGBoost classifier for H/D/A outcome (300 estimators, max_depth=4)
- Two PoissonRegressors for expected goals (home / away)
- Features: ELO diff, squad strength diff, venue flag, rolling 10-game form (pts, GF, GA, GD, win rate) for both teams
- Tournament weights: WC × 3, qualifiers / continental × 2, Nations League × 1.5, friendlies × 0.5

### 4. Ensemble
- **Global weight** set by inverse log-loss on 2,540 held-out test matches (2024–2026):
  - DC: 59.0% accuracy, log-loss 0.88 → **54.9%**
  - ML: 53.7% accuracy, log-loss 1.07 → **45.1%**
- **Per-team weight** for every WC2026 team computed from their individual accuracy on the test set using `x/(x+y)` formula (24–46 games per team — far more reliable than a single tournament)
- Match weight = average of the two teams' per-team weights
- Scoreline matrix: DC matrix rescaled so P(H/D/A) totals match ensemble outcome probs

---

## Data

Real international results from [martj42/international_results](https://github.com/martj42/international_results) (`results.csv`).

- **Training:** 2020-01-01 – 2023-12-31 (3,485 matches)
- **Validation / weight-setting:** 2024-01-01 – 2026-06-09 (2,540 matches)
- **Final model:** trained on full 6-year window before predictions

---

## Running

```bash
pip install numpy pandas scipy scikit-learn xgboost matplotlib
python main.py
```

Outputs:
- Printed table of all 72 group-stage predictions (3 matchdays)
- Sport5 strategy section — games where an underdog has higher EV than the favourite
- 24 pie-chart PNGs saved to `charts/`
- 24 scoreline heat-map PNGs saved to `charts/`

---

## Entering sport5 odds

Open `data.py` and fill in `SPORT5_ODDS` from what you see in the app before each round:

```python
SPORT5_ODDS = {
    ("Mexico", "South Africa"): {"H": 2.0, "D": 4.0, "A": 6.0},
    ...
}
```

- `"H"` = team1 wins, `"D"` = draw, `"A"` = team2 wins
- The lookup handles reversed fixture order automatically
- Leave entries out for rounds with no odds yet — those show `N/A*` in the table

---

## Session log — 2026-06-10

### Real data integration
- Replaced synthetic training data with 6 years of real international results (`results.csv`)
- Team name mapping: `United States → USA`, `Czech Republic → Czechia`
- ELO now initialised from match history (all teams start at 1500, updated chronologically) instead of FIFA rankings
- Scoreline variety improved significantly — recommendations now include 1-0, 2-0, 0-1, 1-1, 0-0 instead of mostly 2-1

### ELO K tuning
- `tune_elo_k()` added — optimises K via Brier-score minimisation on a held-out 12-month window
- Converged to K ≈ 41 on 6-year data (close to the standard 40, confirming the formula is reasonable)

### Data-driven ensemble weights
- Replaced hardcoded 55/45 DC/ML split with validation-derived weights
- **Global:** inverse log-loss over 2,540 test matches → DC 54.9% / ML 45.1%
- **Per-team:** each WC2026 team gets its own DC/ML split based on `accuracy_dc / (accuracy_dc + accuracy_ml)` across their 24–46 individual test games
- Teams like Ecuador and Canada get ~55–59% ML weight (form features predict them better); teams like Curaçao and Paraguay get ~61–65% DC weight (Poisson model handles them better)

### Validation protocol
- Train/test split: 2020–2023 train, 2024–2026 test (proper temporal hold-out, no data leakage)
- Final model retrained on full 6-year window using weights from validation
- `validate_ensemble_weights()` in `models.py` is reusable — pass any train/test split
