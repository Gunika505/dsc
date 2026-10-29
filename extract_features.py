import csv
import re
import sys

FIELDS = [
    "event",
    "game_id",
    "white_id",
    "black_id",
    "result",        
    "date",           
    "time",
    "white_elo",
    "black_elo",
    "rating_diff",
    "eco",
    "time_control",
    "base_seconds",
    "increment",
    "termination",
    "total_moves",
    "num_checks",
    "num_captures",
    "elo_gap",
    "ends_in_checkmate",
    "opening",
]

HEADER_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')
COMMENT_RE = re.compile(r"\{[^}]*\}")          # {[%clk 0:03:00]} etc in move text
VARIATION_RE = re.compile(r"\([^)]*\)")        # not there in this dataset, but just in case
MOVE_NUM_RE = re.compile(r"^\d+\.(\.\.)?$")    # "12." or "12..."
RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}

def clean_int(x):
    try:
        return str(int(x))
    except (TypeError, ValueError):
        return ""


def parse_time_control(tc):
    parts=(tc or "").split("+")
    if len(parts)==2 and parts[0].isdigit() and parts[1].isdigit():
        return parts[0],parts[1]
    return "", "" #missing or invalid time control


def analyse_moves(movetext):
    text=COMMENT_RE.sub(" ",movetext)
    text=VARIATION_RE.sub(" ",text)
    moves=[]
    for token in text.split():
        if token in RESULTS or MOVE_NUM_RE.match(token) or token.startswith("$"):
            continue
        moves.append(token)
    plies = len(moves)
    checks = sum(1 for m in moves if "+" in m or "#" in m)
    captures = sum(1 for m in moves if "x" in m)
    mate = int(plies > 0 and moves[-1].endswith("#"))
    return (plies + 1)//2, checks, captures, mate


def build_row(h, movetext):
    site = h.get("Site", "")
    game_id = site.rsplit("/", 1)[-1] if site else ""
    base, inc = parse_time_control(h.get("TimeControl", ""))
    w_elo, b_elo = clean_int(h.get("WhiteElo")), clean_int(h.get("BlackElo"))
    gap = str(int(w_elo) - int(b_elo)) if w_elo and b_elo else ""
    n_moves, checks, captures, mate = analyse_moves(movetext)
    return [
        h.get("Event", ""), game_id, h.get("White", ""), h.get("Black", ""),
        h.get("Result", ""), h.get("UTCDate", ""), h.get("UTCTime", ""),
        w_elo, b_elo, clean_int(h.get("WhiteRatingDiff")),
        h.get("ECO", ""), h.get("TimeControl", ""), base, inc,
        h.get("Termination", ""), n_moves, checks, captures,
        gap, mate, h.get("Opening", ""),
    ]


def main(src, dst):
    n = 0
    with open(src, "r", encoding="utf-8", errors="replace") as fin, open(dst, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(FIELDS)
        headers, moves = {}, []
        def flush():
            nonlocal n
            if headers:
                writer.writerow(build_row(headers, " ".join(moves)))
                n += 1
                if n%100000==0:
                    print(f"{n:,} games processed...",flush=True)

        for line in fin:
            line=line.strip()
            if not line:
                continue
            if line.startswith("["):
                m=HEADER_RE.match(line)
                if not m:
                    continue
                if m.group(1)=="Event":      #new game
                    flush()
                    headers, moves = {}, []
                headers[m.group(1)] = m.group(2)
            else:
                moves.append(line)
        flush()                                # last game

    print(f"Done. {n:,} games written to {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python extract_features.py <input.pgn> <output.csv>")
    main(sys.argv[1], sys.argv[2])
