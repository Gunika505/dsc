"""
Clean + encode the extracted Lichess CSV into a fully numeric, model-ready table.

Usage:
    python encode_features.py games_2014_04.csv games_encoded.csv
    python encode_features.py games_2014_04.csv games_encoded.csv --top-k 40 --drop-first

Input : the CSV written by extract_features.py
Output: all-numeric CSV (no strings, no missing values) + <output>_eco_top.json
"""
import argparse
import json

import numpy as np
import pandas as pd


def step(df, msg, before):
    print(f"  {msg:<48} -{before - len(df):>9,} rows -> {len(df):>10,} left")
    return len(df)


def main(src, dst, top_k, drop_first):
    df = pd.read_csv(src, dtype={"game_id": str, "eco": str})
    print(f"Raw: {df.shape[0]:,} rows x {df.shape[1]} columns\n\nCleaning:")
    n = len(df)

    # ---------------- minimal cleaning (needed before encoding) ----------------
    df = df.drop_duplicates(subset="game_id");                          n = step(df, "duplicate game_id", n)

    for c in ["white_elo", "black_elo", "rating_diff", "base_seconds", "increment",
              "total_moves", "num_checks", "num_captures"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["white_elo", "black_elo"]);                  n = step(df, "missing Elo ('?')", n)
    df = df.dropna(subset=["base_seconds", "increment"]);               n = step(df, "no clock (correspondence, TimeControl '-')", n)
    df = df[df["total_moves"] > 0];                                     n = step(df, "games with no moves (truncated)", n)
    df = df[df["result"].isin(["1-0", "0-1", "1/2-1/2"])];              n = step(df, "unfinished / invalid result", n)
    df = df[df["eco"].fillna("").str.fullmatch(r"[A-E]\d\d")];          n = step(df, "missing / invalid ECO", n)
    df["rating_diff"] = df["rating_diff"].fillna(0)   # only absent on a few unrated edge cases

    ts = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S", errors="coerce")
    df = df[ts.notna()]; ts = ts[ts.notna()];                           n = step(df, "unparseable date/time", n)
    df = df.reset_index(drop=True); ts = ts.reset_index(drop=True)

    # ---------------- target: result wrt White ----------------
    out = pd.DataFrame(index=df.index)
    out["white_score"] = df["result"].map({"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0})  # 1 win, .5 draw, 0 loss
    out["white_win"] = (out["white_score"] == 1.0).astype("int8")                     # handy for proportion tests

    # ---------------- ratings ----------------
    out["white_elo"] = df["white_elo"].astype(int)
    out["black_elo"] = df["black_elo"].astype(int)
    out["elo_gap"] = out["white_elo"] - out["black_elo"]            # White minus Black
    out["abs_elo_gap"] = out["elo_gap"].abs()
    out["mean_elo"] = (out["white_elo"] + out["black_elo"]) / 2
    out["white_higher_rated"] = (out["elo_gap"] > 0).astype("int8")
    out["white_rating_diff"] = df["rating_diff"].astype(int)        # LEAKAGE: computed after the game

    # ---------------- clock ----------------
    out["base_seconds"] = df["base_seconds"].astype(int)
    out["increment"] = df["increment"].astype(int)
    out["est_duration_s"] = out["base_seconds"] + 40 * out["increment"]   # Lichess' own speed formula
    out["is_tournament"] = df["event"].str.contains("tournament", case=False).astype("int8")

    # ---------------- game content (moves already reduced to counts) ----------------
    out["total_moves"] = df["total_moves"].astype(int)
    out["num_checks"] = df["num_checks"].astype(int)
    out["num_captures"] = df["num_captures"].astype(int)
    out["checks_per_move"] = out["num_checks"] / out["total_moves"]
    out["captures_per_move"] = out["num_captures"] / out["total_moves"]
    out["ends_in_checkmate"] = df["ends_in_checkmate"].astype("int8")

    # ---------------- time of play ----------------
    out["day_of_month"] = ts.dt.day
    out["weekday"] = ts.dt.weekday                       # Mon=0 ... Sun=6
    out["hour"] = ts.dt.hour
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)   # cyclic: 23h is next to 0h
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)

    # ---------------- players: IDs dropped, replaced by activity (frequency encoding) ----------------
    games_per_player = pd.concat([df["white_id"], df["black_id"]]).value_counts()
    out["white_games_played"] = df["white_id"].map(games_per_player).astype(int)
    out["black_games_played"] = df["black_id"].map(games_per_player).astype(int)

    # ---------------- opening: opening name dropped, ECO encoded 4 ways ----------------
    out["eco_number"] = df["eco"].str[1:].astype(int)                      # 0-99 within the letter
    out["eco_freq"] = df["eco"].map(df["eco"].value_counts(normalize=True))  # popularity of the opening
    eco_top = df["eco"].value_counts().head(top_k).index.tolist()
    eco_grp = df["eco"].where(df["eco"].isin(eco_top), "Other")            # top-K codes + "Other"

    # ---------------- one-hot categoricals ----------------
    speed = df["event"].str.extract(r"Rated\s+(\w+)")[0].str.lower().fillna("other")
    cats = pd.DataFrame({
        "speed": speed,                                                     # bullet / blitz / classical ...
        "term": df["termination"].str.replace(" ", "_").str.lower(),        # normal / time_forfeit / ...
        "eco_letter": df["eco"].str[0],                                     # A-E = broad opening family
        "eco": eco_grp,                                                     # top-K ECO codes + Other
    })
    onehot = pd.get_dummies(cats, dtype="int8", drop_first=drop_first)
    out = pd.concat([out, onehot], axis=1)

    # ---------------- sanity + report ----------------
    assert out.isna().sum().sum() == 0, "NaNs left in encoded table"
    assert all(pd.api.types.is_numeric_dtype(t) for t in out.dtypes), "non-numeric column left"

    out.to_csv(dst, index=False)
    with open(dst.rsplit(".", 1)[0] + "_eco_top.json", "w") as f:
        json.dump(eco_top, f)

    r, c = out.shape
    print(f"\nEncoded: {r:,} rows x {c} columns  ->  {dst}")
    print(f"  one-hot columns: {onehot.shape[1]}  (speed, termination, eco letter, top-{top_k} ECO + Other)")
    print(f"  instances x attributes = {r:,} x {c} = {r * c:,}")
    print(f"  >= 3,000,000 ?  {'YES' if r * c >= 3_000_000 else 'NO'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--top-k", type=int, default=30, help="ECO codes kept as their own one-hot column")
    ap.add_argument("--drop-first", action="store_true",
                    help="drop one dummy per category (use for linear/logistic regression)")
    a = ap.parse_args()
    main(a.src, a.dst, a.top_k, a.drop_first)