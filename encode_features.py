"""
cleaning+encoding the CSV from extract_features.py into a table with numbers.
wrt ECO openings, the top-K codes get their own one-hot column and the rest become other

usage:
    python encode_features.py SRC DST [--top-k 30] [--drop-first]

output: all-numeric CSV at DST (no strings/NaNs) + <DST>_eco_top.json (kept ECO codes)
"""

import argparse
import pandas as pd
import json

def main(src, dst, top_k, drop_first):
    def step(df, msg, before):
        print(f"  {msg:<48} -{before - len(df):>9,} rows -> {len(df):>10,} left")
        return len(df)

    df=pd.read_csv(src, dtype={"game_id": str, "eco": str})
    print(f"Raw: {df.shape[0]:,} rows x {df.shape[1]} columns\n\nCleaning:")
    n=len(df)

    #cleaning before encoding
    df=df.drop_duplicates(subset="game_id");                          n=step(df, "duplicate game_id", n)

    for c in ["white_elo", "black_elo", "base_seconds", "increment",
              "total_moves", "num_checks", "num_captures"]:
        df[c]=pd.to_numeric(df[c], errors="coerce")

    df=df.dropna(subset=["white_elo", "black_elo"]);                  n=step(df, "missing Elo ('?')", n)
    df=df.dropna(subset=["base_seconds", "increment"]);               n=step(df, "no clock (correspondence, TimeControl '-')", n)
    df=df[df["total_moves"] > 0];                                     n=step(df, "games with no moves (truncated)", n)
    df=df[df["result"].isin(["1-0", "0-1", "1/2-1/2"])];              n=step(df, "unfinished / invalid result", n)
    df=df[df["eco"].fillna("").str.fullmatch(r"[A-E]\d\d")];          n=step(df, "missing / invalid ECO", n)
    df=df.reset_index(drop=True)

    #target: result wrt white
    encoded=pd.DataFrame(index=df.index)
    encoded["white_score"]=df["result"].map({"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0})  #1 win, .5 draw, 0 loss

    #ratings
    encoded["white_elo"]=df["white_elo"].astype(int)
    encoded["black_elo"]=df["black_elo"].astype(int)
    #white minus black
    encoded["elo_gap"]=encoded["white_elo"] - encoded["black_elo"]            

    #clock
    encoded["base_seconds"]=df["base_seconds"].astype(int)
    encoded["increment"]=df["increment"].astype(int)
    encoded["est_duration_s"]=encoded["base_seconds"] + 40 * encoded["increment"]
    encoded["is_tournament"]=df["event"].str.contains("tournament", case=False).astype("int8")

    #game content
    encoded["total_moves"]=df["total_moves"].astype(int)
    encoded["num_checks"]=df["num_checks"].astype(int)
    encoded["num_captures"]=df["num_captures"].astype(int)
    encoded["ends_in_checkmate"]=df["ends_in_checkmate"].astype("int8")

    #players: IDs dropped, replaced by activity
    games_per_player=pd.concat([df["white_id"], df["black_id"]]).value_counts()
    encoded["white_games_played"]=df["white_id"].map(games_per_player).astype(int)
    encoded["black_games_played"]=df["black_id"].map(games_per_player).astype(int)

    #opening: top-K ECO codes + "other"
    eco_top=df["eco"].value_counts().head(top_k).index.tolist()
    eco_grp=df["eco"].where(df["eco"].isin(eco_top), "Other")

    #one-hot categoricals
    cats=pd.DataFrame({
        "speed": df["event"].str.extract(r"Rated\s+(\w+)")[0].str.lower().fillna("other"),
        "term": df["termination"].str.replace(" ", "_").str.lower(),
        "eco_letter": df["eco"].str[0],     # A-E, broad opening family
        "eco": eco_grp,                     # top-K ECO codes + Other
    })
    onehot=pd.get_dummies(cats, dtype="int8", drop_first=drop_first)
    encoded=pd.concat([encoded, onehot], axis=1)

    #sanity checks+report
    assert encoded.isna().sum().sum() == 0, "NaNs left in encoded table"
    assert all(pd.api.types.is_numeric_dtype(t) for t in encoded.dtypes), "non-numeric column left"

    encoded.to_csv(dst, index=False)
    with open(dst.rsplit(".", 1)[0] + "_eco_top.json", "w") as f:
        json.dump(eco_top, f)

    share=df["eco"].isin(eco_top).mean()
    r, c=encoded.shape
    print(f"\nECO: top {top_k} codes kept, the rest -> 'Other' (top {top_k} cover {share:.1%} of games)")
    print(f"Encoded: {r:,} rows x {c} columns  ->  {dst}")
    print(f"  one-hot columns: {onehot.shape[1]}  (speed, termination, eco letter, top-{top_k} ECO + Other)")
    print(f"  instances x attributes = {r:,} x {c} = {r * c:,}")
    print(f"  >= 3,000,000 ?  {'YES' if r * c >= 3_000_000 else 'NO'}")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--top-k", type=int, default=30, help="ECO codes kept as their own one-hot column")
    parser.add_argument("--drop-first", action="store_true",
                    help="drop one dummy per category (use for linear/logistic regression)")
    a=parser.parse_args()
    main(a.src, a.dst, a.top_k, a.drop_first)
