import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULT_LABELS = {0.0: "Black wins", 0.5: "Draw", 1.0: "White wins"}
OUTLIER_VARS = ["white_elo", "black_elo", "elo_gap", "total_moves", "num_checks" "num_captures", "base_seconds", "increment","white_games_played", "black_games_played"]
CORR_VARS = ["white_score", "white_elo", "black_elo", "elo_gap", "base_seconds",  "increment", "total_moves", "num_checks", "num_captures", "white_games_played", "black_games_played"]

def decode_onehot(frame, stem, skip=()):
    dummy_cols = [c for c in frame.columns if c.startswith(stem) and not c.startswith(tuple(skip))]
    if len(dummy_cols) == 0:
        raise SystemExit(f"Encoded file has no '{stem}*' columns.")
    picked = frame[dummy_cols].idxmax(axis=1).str.slice(len(stem))
    any_set = frame[dummy_cols].sum(axis=1) > 0
    return picked.where(any_set, "baseline")

# calculate mean, count and 95% CI half-width of values for each group in `keys
def summarise(values, keys):
    agg = values.groupby(keys, observed=True).agg(["mean", "count", "std"])
    agg = agg.rename(columns={"count": "n"})
    agg["ci"] = 1.96 * agg["std"] / np.sqrt(agg["n"])
    return agg[["mean", "n", "ci"]]

def finish(fig, folder, fname):
    fig.tight_layout()
    fig.savefig(folder / fname, dpi=150)
    plt.close(fig)
    print("saved file: ", fname)

def heatmap(ax, table, number_fmt, cmap, lo=None, hi=None):
    img = ax.imshow(np.ma.masked_invalid(table.values), cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels(table.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels(table.index)
    for r, c in np.ndindex(table.shape):
        val = table.values[r, c]
        if not np.isnan(val):
            ax.text(c, r, format(val, number_fmt), ha="center", va="center", fontsize=8)
    return img

def eda_elo_distribution(df, order, folder):
    figure, axis = plt.subplots(figsize=(7, 4))
    for col, name in [("white_elo", "White"), ("black_elo", "Black")]:
        axis.hist(df[col], bins=60, alpha=0.6, label=name)
    axis.set(title="Player rating distribution", xlabel="Elo", ylabel="Games")
    axis.legend()
    finish(figure, folder, "01_elo_distribution.png")

def eda_moves_by_speed(df, order, folder):
    figure, axis = plt.subplots(figsize=(7, 4))
    data = [df[df["speed"] == s]["total_moves"] for s in order]
    axis.boxplot(data, flierprops={"marker": ".", "markersize": 1, "alpha": 0.2})
    axis.set_xticks(np.arange(1, len(order) + 1))
    axis.set_xticklabels(order)
    axis.set(title="Game length by speed class", xlabel="Speed class", ylabel="Total moves")
    finish(figure, folder, "02_moves_by_speed_box.png")

def eda_gap_by_result(df, order, folder):
    present = [k for k in RESULT_LABELS if (df["white_score"] == k).any()]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.boxplot([df[df["white_score"] == k]["elo_gap"] for k in present],
                 flierprops={"marker": ".", "markersize": 1, "alpha": 0.2})
    axis.set_xticks(np.arange(1, len(present) + 1))
    axis.set_xticklabels([RESULT_LABELS[k] for k in present])
    axis.axhline(0, color="grey", lw=0.8)
    axis.set(title="Rating gap (White - Black) by result", ylabel="Elo gap")
    finish(figure, folder, "03_elo_gap_by_result_box.png")

def eda_score_by_gap(df, order, folder):
    edges = np.arange(-600, 601, 100)
    buckets = pd.cut(df["elo_gap"].clip(-599, 599), edges)
    stats = summarise(df["white_score"], buckets)
    stats = stats[stats["n"] >= 200]
    centres = [iv.mid for iv in stats.index]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.errorbar(centres, stats["mean"], yerr=stats["ci"], fmt="o-", capsize=3)
    axis.axhline(0.5, color="grey", ls="--", lw=0.8)
    axis.set(title="White's score by rating gap (95% CI)",
             xlabel="Elo gap (White - Black), 100-pt bins", ylabel="Mean White score")
    finish(figure, folder, "04_white_score_by_elo_gap.png")

def eda_score_by_speed(df, order, folder):
    stats = summarise(df["white_score"], df["speed"]).reindex(order)
    low = (stats["mean"] - stats["ci"]).min() - 0.03
    high = (stats["mean"] + stats["ci"]).max() + 0.03
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(stats.index, stats["mean"], yerr=stats["ci"], capsize=4)
    axis.axhline(0.5, color="grey", ls="--", lw=0.8)
    axis.set_ylim(max(0, low), high)
    axis.set(title="White's score by speed class (95% CI; y-axis truncated)",
             ylabel="Mean White score")
    finish(figure, folder, "05_white_score_by_speed.png")

def eda_termination_by_speed(df, order, folder):
    shares = pd.crosstab(df["speed"], df["term"], normalize="index").reindex(order)
    figure, axis = plt.subplots(figsize=(7, 4))
    shares.plot(kind="bar", stacked=True, ax=axis)
    axis.set(title="How games end, by speed class", xlabel="", ylabel="Share of games")
    axis.tick_params(axis="x", rotation=0)
    finish(figure, folder, "06_termination_by_speed.png")

def eda_top_openings(df, order, folder):
    known = df[~df["eco_code"].isin(["Other", "baseline"])]
    top = known["eco_code"].value_counts().head(15).index.tolist()
    stats = summarise(df.loc[df["eco_code"].isin(top), "white_score"],
                      df.loc[df["eco_code"].isin(top), "eco_code"]).reindex(top)
    pos = np.arange(len(top))
    figure, (left, right) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
    left.barh(pos, stats["n"])
    left.set_yticks(pos)
    left.set_yticklabels(top)
    left.invert_yaxis()
    left.set(title="Games played", xlabel="Games")
    right.errorbar(stats["mean"], pos, xerr=stats["ci"], fmt="o", capsize=3)
    right.axvline(0.5, color="grey", ls="--", lw=0.8)
    right.set(title="White's score (95% CI)", xlabel="Mean White score")
    finish(figure, folder, "07_top_openings.png")

def eda_band_by_speed(df, order, folder):
    avg_elo = (df["white_elo"] + df["black_elo"]) / 2
    band = ((avg_elo // 200) * 200).clip(800, 2200).astype(int)
    mean_tbl = df.pivot_table(values="white_score", index=band, columns="speed", aggfunc="mean")[order]
    size_tbl = df.pivot_table(values="white_score", index=band, columns="speed", aggfunc="count")[order]
    mean_tbl = mean_tbl.where(size_tbl >= 500)  # hide thin cells
    mean_tbl.index = [f"{b}+" for b in mean_tbl.index]
    span = np.nanmax(np.abs(mean_tbl.values - 0.5))
    figure, axis = plt.subplots(figsize=(7, 5))
    img = heatmap(axis, mean_tbl, ".3f", "coolwarm", 0.5 - span, 0.5 + span)
    figure.colorbar(img, ax=axis, label="Mean White score")
    axis.set(title="White's score by rating band and speed (cells < 500 games hidden)",
             xlabel="Speed class", ylabel="Mean rating of the two players")
    finish(figure, folder, "08_white_score_band_x_speed.png")

def eda_correlation(df, order, folder):
    corr = df[[c for c in CORR_VARS if c in df.columns]].corr()
    figure, axis = plt.subplots(figsize=(8, 6.5))
    img = heatmap(axis, corr, ".2f", "coolwarm", -1, 1)
    figure.colorbar(img, ax=axis)
    axis.set_title("Correlation of numeric features")
    finish(figure, folder, "09_correlation.png")

PLOTS = [eda_elo_distribution, eda_moves_by_speed, eda_gap_by_result,
         eda_score_by_gap, eda_score_by_speed, eda_termination_by_speed,
         eda_top_openings, eda_band_by_speed, eda_correlation]

def outlier_report(df, folder):
    records = []
    for col in (c for c in OUTLIER_VARS if c in df.columns):
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        spread = q3 - q1
        lower, upper = q1 - 1.5 * spread, q3 + 1.5 * spread
        flagged = int(((df[col] < lower) | (df[col] > upper)).sum())
        records.append({"variable": col, "q1": q1, "q3": q3,
                        "lower_fence": lower, "upper_fence": upper,
                        "n_flagged": flagged,
                        "pct_flagged": round(100 * flagged / len(df), 2)})
    report = pd.DataFrame(records)
    report.to_csv(folder / "outlier_report.csv", index=False)
    print("\nIQR outlier report (flagged, NOT removed):\n", report.to_string(index=False))

def raw_speed_and_score(raw):
    speed = raw["event"].str.extract(r"Rated\s+(\w+)")[0].str.lower().fillna("other")
    score = raw["result"].map({"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0})
    return speed, score

def summary_block(n_rows, n_cols, missing_elo, w_elo, b_elo, moves, score, speed):
    info = {
        "rows": n_rows,
        "columns": n_cols,
        "missing Elo values": missing_elo,
        "mean white_elo": w_elo.mean(),
        "median white_elo": w_elo.median(),
        "mean black_elo": b_elo.mean(),
        "mean total_moves": moves.mean(),
        "median total_moves": moves.median(),
        "White mean score": score.mean(),
    }
    for name, frac in speed.value_counts(normalize=True).items():
        info[f"share {name} (%)"] = 100 * frac
    return info

def before_after_table(raw_path, clean, folder):
    raw = pd.read_csv(raw_path, dtype={"game_id": str, "eco": str})
    for col in ("white_elo", "black_elo", "total_moves"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw_speed, raw_score = raw_speed_and_score(raw)
    def n_missing(t):
        return int(t["white_elo"].isna().sum() + t["black_elo"].isna().sum())

    before = summary_block(len(raw), raw.shape[1], n_missing(raw), raw["white_elo"],
    raw["black_elo"], raw["total_moves"], raw_score, raw_speed)
    after = summary_block(len(clean), clean.shape[1]-3, n_missing(clean), clean["white_elo"], clean["black_elo"], clean["total_moves"], clean["white_score"], clean["speed"])
    table = pd.DataFrame({"before (raw)": before, "after (encoded)": after}).round(4)
    table.to_csv(folder / "before_after.csv")
    print("\nBefore vs after data preparation:\n", table.to_string())

def run(src, raw_path, out_dir):
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)
    df["speed"] = decode_onehot(df, "speed_")
    df["term"] = decode_onehot(df, "term_")
    df["eco_code"] = decode_onehot(df, "eco_", skip=("eco_letter_",))
    order = df.groupby("speed")["est_duration_s"].median().sort_values().index.tolist()
    print(f"Loaded {len(df):,} games. Speed classes: {order}")
    for make_plot in PLOTS:
        make_plot(df, order, folder)
    outlier_report(df, folder)
    if raw_path:
        before_after_table(raw_path, df, folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="encoded CSV ")
    parser.add_argument("--raw", help="raw CSV")
    parser.add_argument("--out", default="eda_plots", help="output folder")
    args = parser.parse_args()
    run(args.src, args.raw, args.out)
