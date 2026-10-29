**Group:** DataMinerz

| # | Member | Roll No. |
|---|--------|----------|
| 1 | Gunika Singh | 2023225 |
| 2 | Aditi Aryan | 2023037 |
| 3 | Navya Gupta | 2023343 |
| 4 | Khushi Rana | 2023285 |
| 5 | Preet Kuthati | 2023398 |

---

## Dataset

The **Lichess Standard Rated Games database for April 2014**: every rated standard-chess game played on [lichess.org](https://lichess.org) that month, stored in PGN format (headers such as players, Elo ratings, time control, ECO opening code and result, followed by the move list).

- **Link:** https://database.lichess.org/
- **File used:** `lichess_db_standard_rated_2014-04.pgn.zst` (zstandard-compressed PGN)

## Files

| File | Purpose |
|------|---------|
| `extract_features.py` | Streams the PGN and writes one row per game (ratings, time control, result, ECO, number of moves, checks, captures, checkmate flag) to a raw CSV. |
| `encode_features.py` | Cleans the raw CSV and turns it into a fully numeric table (one-hot encoding; top-K ECO codes plus "Other"). |
| `eda_plots.py` | Produces 9 plots, an IQR outlier report and a before/after data-preparation table. |
| `final_tests.py` | Runs the five hypothesis tests and writes the results. |

## Order of execution

Run from the project folder, one step at a time. Each step needs the output of the one before it.

### 1. Download the `.zst` file

Download `lichess_db_standard_rated_2014-04.pgn.zst` from https://database.lichess.org/ and put it in the project folder.

### 2. Decompress

```
zstd -d lichess_db_standard_rated_2014-04.pgn.zst
open -a TextEdit "lichess_db_standard_rated_2014-04.pgn"
```

The first command creates `lichess_db_standard_rated_2014-04.pgn`; the second opens it in TextEdit.

### 3. Extract

```
python extract_features.py lichess_db_standard_rated_2014-04.pgn games_raw.csv
```

Output: `games_raw.csv`

### 4. Encode

```
python encode_features.py games_raw.csv games_encoded.csv
```

Output: `games_encoded.csv` and `games_encoded_eco_top.json`

### 5. EDA

```
python eda_plots.py games_encoded.csv --raw games_raw.csv
```

Output (in `eda_plots/`): `01_elo_distribution.png` to `09_correlation.png`, `outlier_report.csv`, `before_after.csv`

### 6. Hypothesis tests

```
python final_tests.py games_encoded.csv
```

Output: `final_results.csv`, `final_results.json`, `results_table.tex`

## All commands in order

```
zstd -d lichess_db_standard_rated_2014-04.pgn.zst
open -a TextEdit "lichess_db_standard_rated_2014-04.pgn"
python extract_features.py lichess_db_standard_rated_2014-04.pgn games_raw.csv
python encode_features.py games_raw.csv games_encoded.csv
python eda_plots.py games_encoded.csv --raw games_raw.csv
python final_tests.py games_encoded.csv
```
