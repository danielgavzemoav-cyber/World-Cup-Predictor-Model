# FIFA World Cup 2026 Predictor

EV-optimised prediction model for the sport5 "5 חבר'ה" game.  
Goal: maximise expected sport5 points, not just predict the most likely result.

**Tournament status:** Group stage complete. Now predicting the Round of 32.

---

## How the sport5 scoring works

- Correct result (W/D/L) → base odds points
- Exact scoreline → base odds **+ 6** bonus points (all rounds)

So the EV of predicting score G1-G2 is:

```
EV = odds[result] × P(result) + 6 × P(exact G1-G2)
```

This means picking a high-odds underdog can beat picking the favourite even if the favourite is more likely to win — especially in the knockout stage where odds spreads are much wider.

---

## Architecture (4 files)

| File | Role |
|------|------|
| `data.py` | 48 teams, 12 groups, 72 group fixtures, 16 R32 knockout fixtures, `SPORT5_ODDS` (all rounds) |
| `models.py` | ELO system, Dixon-Coles, XGBoost ML, Ensemble |
| `predict.py` | EV computation, group + knockout predictors, strategy printer, charts |
| `main.py` | Orchestration — runs everything end-to-end |

---

## Models

### 1. Dynamic ELO
- Initialised from match history, updated across 6 years of real international results
- K parameter **tuned automatically** via Brier-score minimisation on a held-out window — converged to K ≈ 41
- Goal-difference multiplier: `1 + 0.5 × log1p(|g1-g2|)`
- All 72 group-stage results fed in post-training so knockout predictions reflect actual tournament form

### 2. Dixon-Coles (DC)
- Bivariate Poisson with ρ-correction for low-score correlations (0-0, 1-0, 0-1, 1-1)
- Attack/defence parameters fitted via NLL optimisation on real data
- λ values scaled by a dynamic ELO weight that grows with the ELO gap between teams
- Home advantage only for Mexico, Canada, USA (tournament hosts)

### 3. ML (XGBoost)
- XGBoost classifier for H/D/A outcome; two Poisson regressors for xG
- Features: ELO diff, squad strength diff, venue flag, 10-game rolling form (pts, GF, GA, GD, win rate) for both teams
- Currently weighted at 0% in the ensemble (DC-only mode) — ML was over-crediting underdogs

### 4. Ensemble
- **Global weight** derived from inverse log-loss on 2,564 held-out test matches (2024–2026):
  - DC: 58.9% accuracy → **54.9%** global weight
  - ML: 53.6% accuracy → **45.1%** global weight
- **Per-team weight** for every WC2026 team from their individual accuracy using `x/(x+y)` formula
- Match weight = average of the two teams' per-team weights
- **Active override:** 100% DC in `main.py` — ML over-credits underdogs in this tournament

---

## Data

Real international results from [martj42/international_results](https://github.com/martj42/international_results) (`results.csv`).

- **Training window:** 2020-01-01 – 2023-12-31 (3,485 matches)
- **Validation / weight-setting:** 2024-01-01 – 2026-06-17 (2,564 matches)
- **Final model:** retrained on the full 6-year window before predictions

---

## Running

```bash
pip install numpy pandas scipy scikit-learn xgboost matplotlib
python main.py
```

Outputs:
- Printed table of all 72 group-stage predictions (3 matchdays)
- Printed table of all 16 Round-of-32 knockout predictions
- Sport5 strategy section for both group stage and knockout — games where an underdog has higher EV
- 24 pie-chart PNGs saved to `charts/`
- 24 scoreline heat-map PNGs saved to `charts/`

---

## Entering sport5 odds

Open `data.py` and fill in `SPORT5_ODDS` from the app before each round:

```python
SPORT5_ODDS = {
    ("Mexico", "South Africa"): {"H": 2.0, "D": 4.0, "A": 6.0},
    # knockout:
    ("Canada", "South Africa"): {"H": 4, "D": 7, "A": 9},
    ...
}
```

- `"H"` = team1 wins, `"D"` = draw, `"A"` = team2 wins
- The lookup handles reversed fixture order automatically
- Leave entries out for rounds with no odds yet — those show `N/A*` in the table
- For knockout rounds beyond R32, also add a `KNOCKOUT_Rxx_FIXTURES` list and a new predict call in `main.py`

---

## Adding a new knockout round

1. Add the new fixtures list to `data.py`:
   ```python
   KNOCKOUT_R16_FIXTURES = [("Winner A", "Winner B"), ...]
   ```
2. Add odds to `SPORT5_ODDS` in `data.py`
3. In `predict.py`, update `predict_knockout_stage()` to iterate over the new fixtures list instead of `KNOCKOUT_R32_FIXTURES` (or duplicate the function for the new round)
4. Call it in `main.py` after the R32 block:
   ```python
   ko16 = predict_knockout_stage(ensemble, elo, matches)
   print_knockout_summary(ko16)
   print_sport5_strategy(ko16, exact_bonus=SPORT5_EXACT_BONUS_KNOCKOUT, title="R16 STRATEGY")
   ```
5. Run `python main.py`
