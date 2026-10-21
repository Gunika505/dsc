"""
Extract game-level features from a Lichess PGN dump into a CSV.

Usage:
    python extract_features.py lichess_db_standard_rated_2014-04.pgn games_2014_04.csv
    (also works on .pgn.bz2 / .pgn.zst directly, no need to decompress)

Streams the file line by line, so memory use stays small even for multi-GB dumps.
"""
import csv
import re
import sys
import bz2

FIELDS = [
    "event",          # Rated Blitz / Bullet / Classical ... game
    "game_id",        # from the Site URL
    "white_id",
    "black_id",
    "result",         # 1-0, 0-1, 1/2-1/2
    "date",           # UTCDate
    "time",           # UTCTime
    "white_elo",
    "black_elo",
    "rating_diff",    # WhiteRatingDiff header: Elo points White gained/lost from this game
    "eco",
    "time_control",   # raw, e.g. 180+0
    "base_seconds",   # parsed from time_control
    "increment",      # parsed from time_control
    "termination",    # Normal / Time forfeit / Abandoned / Rules infraction
    "total_moves",    # full moves (ceil(plies / 2))
    "num_checks",     # moves with + or #
    "num_captures",   # moves with x
    # extras (drop if you don't need them)
    "elo_gap",        # white_elo - black_elo
    "ends_in_checkmate",
    "opening",
]

HEADER_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')
COMMENT_RE = re.compile(r"\{[^}]*\}")          # {[%clk 0:03:00]} etc.
VARIATION_RE = re.compile(r"\([^)]*\)")        # rare, but strip anyway
MOVE_NUM_RE = re.compile(r"^\d+\.(\.\.)?$")    # "12." or "12..."
RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}


def open_any(path):
    if path.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    if path.endswith(".zst"):
        import zstandard  # pip install zstandard
        import io
        fh = open(path, "rb")
        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh),
                                encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def clean_int(x):
    """'1806' -> '1806', '?' or '' -> ''  (missing)."""
    x = (x or "").strip().lstrip("+")
    return x if re.fullmatch(r"-?\d+", x) else ""


def parse_time_control(tc):
    m = re.fullmatch(r"(\d+)\+(\d+)", tc or "")
    if m:
        return m.group(1), m.group(2)
    return "", ""      # "-" (correspondence) or missing


def analyse_moves(movetext):
    text = COMMENT_RE.sub(" ", movetext)
    text = VARIATION_RE.sub(" ", text)
    moves = []
    for tok in text.split():
        if tok in RESULTS or MOVE_NUM_RE.match(tok) or tok.startswith("$"):
            continue
        moves.append(tok)
    plies = len(moves)
    checks = sum(1 for m in moves if "+" in m or "#" in m)
    captures = sum(1 for m in moves if "x" in m)
    mate = int(plies > 0 and moves[-1].endswith("#"))
    return (plies + 1) // 2, checks, captures, mate


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
    with open_any(src) as fin, open(dst, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(FIELDS)

        headers, moves = {}, []

        def flush():
            nonlocal n
            if headers:
                writer.writerow(build_row(headers, " ".join(moves)))
                n += 1
                if n % 100_000 == 0:
                    print(f"{n:,} games processed...", flush=True)

        for line in fin:
            line = line.strip()
            if not line:
                continue
            if line.startswith("["):
                m = HEADER_RE.match(line)
                if not m:
                    continue
                if m.group(1) == "Event":      # a new game starts
                    flush()
                    headers, moves = {}, []
                headers[m.group(1)] = m.group(2)
            else:
                moves.append(line)             # movetext (may span lines)
        flush()                                # last game

    print(f"Done. {n:,} games written to {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python extract_features.py <input.pgn> <output.csv>")
    main(sys.argv[1], sys.argv[2])