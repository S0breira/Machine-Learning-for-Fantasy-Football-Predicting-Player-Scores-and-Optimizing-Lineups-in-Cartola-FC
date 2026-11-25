# Fantasy Football in the Brasileirão: Forecasting Player Points and Building Budget-Constrained Lineups in Cartola FC

Final project for the Artificial Intelligence course of the Computer Science program at the Federal University of Piauí (UFPI).

The goal is to predict how many points each player will score in Cartola FC, the official fantasy football game of the Brazilian Série A (Brasileirão), and to use these predictions to pick the best possible lineup each round within the player's budget.

## Contents
- `notebook/cartola_fc_ml.ipynb`: the complete project, with explanations, code and results, in Portuguese. It runs from start to finish, downloading the data by itself.
- `src/`: the same pipeline as Python scripts.

## Data
- **Cartola FC (2018–2026)**: public dataset collected by the caRtola project, https://github.com/henriquepgomide/caRtola (MIT license). A fixed version of the repository is used, so the data do not change.
- **Match results**: football-data.co.uk, https://www.football-data.co.uk/brazil.php (file `BRA.csv`). It is used only to identify home/away games and to validate the inferred opponents. Matches are cut at 2026-11-14, the date of the data used in this work, so the results can be reproduced.

The data is not stored in this repository. It is downloaded to `data/raw` by the notebook or by `src/baixar_dados.py`.

## Method
1. **Data cleaning**:
   - unifies the file formats of all seasons;
   - converts cumulative scouts into per-round values and fixes corrupted cumulative files;
   - removes information leakage: prices, status and averages published after each round;
   - rebuilds rounds with missing scores;
   - infers each team's opponent from the scouts.
2. **Features**:
   - recent form weighted in blocks of 5 games;
   - inferred starting status;
   - tactical roles (K-means);
   - Elo ratings of the clubs;
   - penalty-taker estimates (softmax with split credit);
   - promoted clubs;
   - a price rule.
3. **Model**: LightGBM regression of the player's points (0 if the player does not play). The split is temporal:
   - training: 2018–2024;
   - validation: 2025, rounds 1–19;
   - test: 2025, rounds 20–38;
   - holdout: 2026;
   - each period is predicted by a model trained only on earlier data.
4. **Lineup optimization**: integer programming (`scipy.optimize.milp`). It handles:
   - the official formations, the coach and the captain (1.5×);
   - the budget, which changes with player prices;
   - substitutes and the luxury substitute;
   - a limit on players from the same club.
5. **Baseline**: the same optimizer, picking players by their Cartola average.

## Results (season simulations, starting budget C$ 100)

| | Model 2024 | Model 2025 | Model 2026 | Baseline 2024 | Baseline 2025 | Baseline 2026 |
|---|---|---|---|---|---|---|
| Points per round | **65.0** | **71.3** | **74.1** | 49.3 | 59.9 | 53.0 |
| Final budget (C$) | 99.3 | 96.8 | 121.1 | 86.1 | 118.5 | 95.8 |
| Starters without replacement | 10 | 8 | 7 | 17 | 22 | 15 |

## How to run
Notebook: open `notebook/cartola_fc_ml.ipynb` in Google Colab or Jupyter and run all cells (about 5 minutes).

Scripts:
```
pip install -r requirements.txt
cd src
python executar_tudo.py
```
This downloads the data (if needed) and runs the whole pipeline. Outputs are written to `data/processed`.

## Files (`src/`)
| File | Purpose |
|---|---|
| `baixar_dados.py` | Downloads the raw data |
| `base_unificada.py` | Data cleaning, leakage removal, opponent inference |
| `atributos.py` | Basic player history |
| `ponderado.py` | Weighted averages in blocks of 5 games |
| `penalti.py` | Penalty-taker model |
| `atributos_final.py` | All model features and the baseline prediction |
| `modelo_final.py` | LightGBM training, ablation and comparison with the baseline |
| `otimizador.py` | Lineup optimization and season simulation |
| `executar_tudo.py` | Runs the full pipeline |
