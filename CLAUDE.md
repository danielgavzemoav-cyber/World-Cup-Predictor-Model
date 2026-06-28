# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## GitHub sync reminder

Every 30 minutes of active work, ask the user: "Want me to commit and push the current changes to GitHub?"  
Only push if they say yes. Never push automatically.

## Running the model

```bash
python main.py          # full run: ~110s, trains everything, prints all 72 group-stage + 16 R32 predictions
```

No test suite or linter is configured. Verify changes by running `main.py` and checking the prediction output looks sane.

## What the model does

Predicts all 72 FIFA WC2026 group-stage games and all 16 Round-of-32 knockout fixtures, recommending sport5 "5 חבר'ה" scorelines that **maximise expected points**, not just most-probable outcome. The sport5 scoring is:

- Correct result → `odds` pts
- Exact score → `odds + 6` pts (all rounds — group stage and knockout)

So `EV(predict G1-G2) = odds[result] × P(result) + 6 × P(exact G1-G2)`.

## File responsibilities

- **`data.py`** — all static data: `WC2026_GROUPS` (12 groups, 48 teams), `ALL_GROUP_FIXTURES` (72 games), `KNOCKOUT_R32_FIXTURES` (16 R32 matchups), `HOST_NATIONS`, `SQUAD_STRENGTH`, `TEAM_NAME_MAP` (normalises `results.csv` names), `SPORT5_ODDS` (all rounds, fill in from the app before each round), `SPORT5_EXACT_BONUS_GROUP = 6`, `SPORT5_EXACT_BONUS_KNOCKOUT = 6`.
- **`models.py`** — everything trainable: `ELOSystem`, `DixonColesModel`, `MLModel`, `EnsembleModel`, plus `load_real_data()`, `tune_elo_k()`, `validate_ensemble_weights()`, `build_ml_features()`.
- **`predict.py`** — inference only: `predict_group_stage()`, `print_group_stage_summary()`, `predict_knockout_stage()`, `print_knockout_summary()`, `print_sport5_strategy(exact_bonus, title)`, `_get_odds()` (handles reversed fixture order), `optimal_prediction()` (picks max-EV score), chart functions.
- **`main.py`** — orchestration. Contains two top-level toggles (`GENERATE_PIES`, `GENERATE_HEATMAPS`) and two date constants (`DATA_MIN_DATE = "2020-01-01"`, `VALIDATION_SPLIT = "2024-01-01"`). Ensemble is overridden to 100% DC / 0% ML (ML over-credits underdogs).
- **`results.csv`** — martj42/international_results dataset (≈49k rows from 1872; only rows ≥ `DATA_MIN_DATE` are used).

## Pipeline order (main.py)

1. `load_real_data()` — loads `results.csv`, applies `TEAM_NAME_MAP`, maps `neutral` → `venue`
2. Train/test split at `VALIDATION_SPLIT`
3. `tune_elo_k()` — optimises ELO K via Brier score on last 12 months of **train** data only
4. `validate_ensemble_weights()` — trains fresh DC+ML on train split, evaluates on test split, returns global inverse-log-loss weights + per-team `x/(x+y)` accuracy weights
5. `ELOSystem.fit_from_history()` on full data with tuned K
6. `DixonColesModel.fit()` on full data
7. `build_ml_features()` + `MLModel.train()` on full data
8. `EnsembleModel(dc, ml, dc_weight=1.0, team_weights={})` — 100% DC override
9. `apply_actual_results(elo)` — feeds all 72 real WC2026 group-stage results into `elo.ratings` (after training, so historical ML/DC features aren't contaminated by post-tournament ratings)
10. `predict_group_stage()` → `print_group_stage_summary()` → `print_sport5_strategy()`
11. `predict_knockout_stage()` → `print_knockout_summary()` → `print_sport5_strategy(exact_bonus=SPORT5_EXACT_BONUS_KNOCKOUT)`

## Key design decisions

**ELO** is the single source of team strength for both DC λ-scaling and ML features. It is recomputed from scratch each run (no persistence). The `update()` method uses a goal-difference multiplier (`1 + 0.5 × log1p(|g1-g2|)`). In-tournament ELO updates flow into round 2/3 predictions via `elo.snapshot()` / `elo.restore()`.

**Dixon-Coles λ scaling** uses a dynamic ELO weight that grows with ELO gap:
```python
dyn_w = min(0.85, elo_weight + abs(e1 - e2) / 1200.0)
l1_dc *= (e1 / avg) ** dyn_w
```
This makes strong-vs-weak mismatches produce realistic xG (e.g. Germany vs Curaçao ~2.5–0.8).

**ML features** (`FEATURE_COLS` in models.py): `elo_diff`, `squad_diff`, `venue_home`, plus 10-game rolling form (pts, GF, GA, GD, win rate) for both teams. `rank_diff` was removed — ELO now covers that signal. Form is precomputed in O(n log n) via `_precompute_form()`; avoid calling `_team_form()` in a loop over many matches (it's O(n) per call).

**Ensemble per-team weights**: stored in `EnsembleModel.team_weights` as `{team: {"dc": w, "ml": 1-w, "source": "..."}}`. For a match, `_match_weights(t1, t2)` averages the two teams' DC weights. Teams absent from the test set fall back to the global weight. **Currently overridden to 100% DC** in main.py because ML over-credits underdogs.

**Sport5 odds lookup**: `_get_odds(t1, t2)` in `predict.py` checks both `(t1,t2)` and `(t2,t1)` in `SPORT5_ODDS`, swapping H/A when the reversed key is found.

**Knockout vs group stage**: The model uses the same EV formula for both. The only structural differences are: (1) `predict_knockout_stage()` passes `exact_bonus=SPORT5_EXACT_BONUS_KNOCKOUT` to `optimal_prediction()`; (2) ELO ratings used include all 72 group-stage results; (3) `as_of` date is `2026-06-28`. No ET/penalties modeling — sport5 asks for 90-minute predictions.

## Updating for a new round

1. Open `data.py`, fill in `SPORT5_ODDS` with the odds shown in the sport5 app for that round.
2. Append that round's real scores to `actual_results` in `apply_actual_results()` (`main.py`) — this feeds `elo.ratings` so later-round predictions reflect actual in-tournament form.
3. For knockout rounds beyond R32, add fixtures to a new `KNOCKOUT_Rxx_FIXTURES` list in `data.py` and a corresponding `predict_knockout_stage()` call in `main.py`.
4. Run `python main.py`.
