"""
EDA plots and tables for the encoded Lichess data.

Usage:
    python eda_plots.py games_encoded.csv
    python eda_plots.py games_encoded.csv --raw games_2014_04.csv    # also builds the before/after table

Needs: pandas, numpy, matplotlib
Output (folder eda_plots/ by default):
    01_elo_distribution.png          06_termination_by_speed.png
    02_moves_by_speed_box.png        07_top_openings.png
    03_elo_gap_by_result_box.png     08_white_score_band_x_speed.png
    04_white_score_by_elo_gap.png    09_correlation.png
    05_white_score_by_speed.png      outlier_report.csv, before_after.csv (with --raw)
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "eda_plots"


# ------------------------------------------------------------------ helpers
def undo_onehot(df, prefix, exclude=()):
    """Rebuild a single categorical column from its one-hot columns."""
    cols = [c for c in df.columns if c.startswith(prefix) and not any(c.startswith(e) for e in exclude)]
    if not cols:
        raise SystemExit(f"No columns starting with '{prefix}' found in the encoded file.")
    labels = df[cols].idxmax(axis=1).str[len(prefix):]
    return labels.where(df[cols].sum(axis=1) > 0, "baseline")   # all-zero row = dropped dummy (--drop-first)


def group_stats(values, groups):
    g = values.groupby(groups, observed=True)
    out = pd.DataFrame({"mean": g.mean(), "n": g.count()})
    out["ci"] = 1.96 * g.std(ddof=1) / np.sqrt(out["n"])        # normal-approx 95% CI
    return out


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)
    print("saved", name)


def annotated_heatmap(ax, mat, fmt, cmap, vmin=None, vmax=None):
    im = ax.imshow(np.ma.masked_invalid(mat.values), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8)
    return im


# ------------------------------------------------------------------ plots
def main(src, raw):
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(src)
    df["speed"] = undo_onehot(df, "speed_")
    df["term"] = undo_onehot(df, "term_")
    df["eco_code"] = undo_onehot(df, "eco_", exclude=("eco_letter_",))
    speed_order = df.groupby("speed")["est_duration_s"].median().sort_values().index.tolist()
    print(f"Loaded {len(df):,} games. Speed classes: {speed_order}")

    # 1. Elo distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["white_elo"], bins=60, alpha=0.6, label="White")
    ax.hist(df["black_elo"], bins=60, alpha=0.6, label="Black")
    ax.set(title="Player rating distribution", xlabel="Elo", ylabel="Games"); ax.legend()
    save(fig, "01_elo_distribution.png")

    # 2. Game length by speed (boxplot, outliers shown on purpose)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([df.loc[df["speed"] == s, "total_moves"] for s in speed_order],
               flierprops=dict(marker=".", markersize=1, alpha=0.2))
    ax.set_xticks(range(1, len(speed_order) + 1)); ax.set_xticklabels(speed_order)
    ax.set(title="Game length by speed class", xlabel="Speed class", ylabel="Total moves")
    save(fig, "02_moves_by_speed_box.png")

    # 3. Elo gap by result (boxplot)
    names = {0.0: "Black wins", 0.5: "Draw", 1.0: "White wins"}
    keys = [k for k in (0.0, 0.5, 1.0) if (df["white_score"] == k).any()]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([df.loc[df["white_score"] == k, "elo_gap"] for k in keys],
               flierprops=dict(marker=".", markersize=1, alpha=0.2))
    ax.set_xticks(range(1, len(keys) + 1)); ax.set_xticklabels([names[k] for k in keys])
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(title="Rating gap (White - Black) by result", ylabel="Elo gap")
    save(fig, "03_elo_gap_by_result_box.png")

    # 4. White score vs Elo-gap bin
    bins = np.arange(-600, 601, 100)
    gap_bin = pd.cut(df["elo_gap"].clip(-599, 599), bins)
    st = group_stats(df["white_score"], gap_bin)
    st = st[st["n"] >= 200]
    x = [iv.mid for iv in st.index]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(x, st["mean"], yerr=st["ci"], fmt="o-", capsize=3)
    ax.axhline(0.5, color="grey", ls="--", lw=0.8)
    ax.set(title="White's score by rating gap (95% CI)", xlabel="Elo gap (White - Black), 100-pt bins",
           ylabel="Mean White score")
    save(fig, "04_white_score_by_elo_gap.png")

    # 5. White score by speed class
    st = group_stats(df["white_score"], df["speed"]).loc[speed_order]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(st.index, st["mean"], yerr=st["ci"], capsize=4)
    ax.axhline(0.5, color="grey", ls="--", lw=0.8)
    ax.set_ylim(max(0, (st["mean"] - st["ci"]).min() - 0.03), (st["mean"] + st["ci"]).max() + 0.03)
    ax.set(title="White's score by speed class (95% CI; y-axis truncated)", ylabel="Mean White score")
    save(fig, "05_white_score_by_speed.png")

    # 6. Termination mix by speed
    ct = pd.crosstab(df["speed"], df["term"], normalize="index").loc[speed_order]
    fig, ax = plt.subplots(figsize=(7, 4))
    ct.plot(kind="bar", stacked=True, ax=ax)
    ax.set(title="How games end, by speed class", xlabel="", ylabel="Share of games")
    ax.tick_params(axis="x", rotation=0)
    save(fig, "06_termination_by_speed.png")

    # 7. Top openings: popularity and White score
    top = df.loc[~df["eco_code"].isin(["Other", "baseline"]), "eco_code"].value_counts().head(15).index.tolist()
    sub = df[df["eco_code"].isin(top)]
    st = group_stats(sub["white_score"], sub["eco_code"]).loc[top]
    y = np.arange(len(top))
    fig, (a1, a2) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
    a1.barh(y, st["n"]); a1.set_yticks(y); a1.set_yticklabels(top); a1.invert_yaxis()
    a1.set(title="Games played", xlabel="Games")
    a2.errorbar(st["mean"], y, xerr=st["ci"], fmt="o", capsize=3); a2.axvline(0.5, color="grey", ls="--", lw=0.8)
    a2.set(title="White's score (95% CI)", xlabel="Mean White score")
    save(fig, "07_top_openings.png")

    # 8. White score: rating band x speed class (the 'novel angle' view)
    band = (((df["white_elo"] + df["black_elo"]) / 2) // 200 * 200).clip(800, 2200).astype(int)
    pv = df.pivot_table(values="white_score", index=band, columns="speed", aggfunc="mean")[speed_order]
    cnt = df.pivot_table(values="white_score", index=band, columns="speed", aggfunc="count")[speed_order]
    pv = pv.where(cnt >= 500)                                     # hide thin cells
    pv.index = [f"{b}+" for b in pv.index]
    d = np.nanmax(np.abs(pv.values - 0.5))
    fig, ax = plt.subplots(figsize=(7, 5))
    im = annotated_heatmap(ax, pv, ".3f", "coolwarm", 0.5 - d, 0.5 + d)
    fig.colorbar(im, ax=ax, label="Mean White score")
    ax.set(title="White's score by rating band and speed (cells < 500 games hidden)",
           xlabel="Speed class", ylabel="Mean rating of the two players")
    save(fig, "08_white_score_band_x_speed.png")

    # 9. Correlation heatmap
    cols = [c for c in ["white_score", "white_elo", "black_elo", "elo_gap", "base_seconds", "increment",
                        "total_moves", "num_checks", "num_captures",
                        "white_games_played", "black_games_played"] if c in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = annotated_heatmap(ax, corr, ".2f", "coolwarm", -1, 1)
    fig.colorbar(im, ax=ax)
    ax.set_title("Correlation of numeric features")
    save(fig, "09_correlation.png")

    # IQR outlier report (flags only, nothing is removed)
    rows = []
    for c in ["white_elo", "black_elo", "elo_gap", "total_moves", "num_checks", "num_captures",
              "base_seconds", "increment", "white_games_played", "black_games_played"]:
        if c not in df.columns:
            continue
        q1, q3 = df[c].quantile([0.25, 0.75]); iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[c] < lo) | (df[c] > hi)).sum())
        rows.append({"variable": c, "q1": q1, "q3": q3, "lower_fence": lo, "upper_fence": hi,
                     "n_flagged": n_out, "pct_flagged": round(100 * n_out / len(df), 2)})
    rep = pd.DataFrame(rows)
    rep.to_csv(os.path.join(OUT, "outlier_report.csv"), index=False)
    print("\nIQR outlier report (flagged, NOT removed):\n", rep.to_string(index=False))

    # Before / after table
    if raw:
        before_after(raw, df)


def raw_view(d):
    speed = d["event"].str.extract(r"Rated\s+(\w+)")[0].str.lower().fillna("other")
    score = d["result"].map({"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0})
    return speed, score


def before_after(raw_path, clean):
    raw = pd.read_csv(raw_path, dtype={"game_id": str, "eco": str})
    for c in ["white_elo", "black_elo", "total_moves"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    r_speed, r_score = raw_view(raw)

    def block(n_rows, n_cols, elo_missing, w_elo, b_elo, moves, score, speed):
        d = {"rows": n_rows, "columns": n_cols, "missing Elo values": elo_missing,
             "mean white_elo": w_elo.mean(), "median white_elo": w_elo.median(),
             "mean black_elo": b_elo.mean(), "mean total_moves": moves.mean(),
             "median total_moves": moves.median(), "White mean score": score.mean()}
        for s, share in speed.value_counts(normalize=True).items():
            d[f"share {s} (%)"] = 100 * share
        return d

    before = block(len(raw), raw.shape[1], int(raw["white_elo"].isna().sum() + raw["black_elo"].isna().sum()),
                   raw["white_elo"], raw["black_elo"], raw["total_moves"], r_score, r_speed)
    after = block(len(clean), clean.shape[1], int(clean["white_elo"].isna().sum() + clean["black_elo"].isna().sum()),
                  clean["white_elo"], clean["black_elo"], clean["total_moves"], clean["white_score"], clean["speed"])
    tab = pd.DataFrame({"before (raw)": before, "after (encoded)": after}).round(4)
    tab.to_csv(os.path.join(OUT, "before_after.csv"))
    print("\nBefore vs after data preparation:\n", tab.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="encoded CSV from encode_features.py")
    ap.add_argument("--raw", help="raw CSV from extract_features.py (for the before/after table)")
    ap.add_argument("--out", default="eda_plots", help="output folder")
    a = ap.parse_args()
    OUT = a.out
    main(a.src, a.raw)
