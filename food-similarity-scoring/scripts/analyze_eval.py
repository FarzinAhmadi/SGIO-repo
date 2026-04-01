"""Analyze manual evaluation results from the SQLite database.

Usage:
    uv run python -m scripts.analyze_eval
    uv run python -m scripts.analyze_eval --db data/eval/eval.db
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _subheader(title: str) -> None:
    print(f"\n--- {title} ---")


def _table(headers: list[str], rows: list[list], col_widths: list[int] | None = None) -> None:
    if not rows:
        print("  (no data)")
        return
    if col_widths is None:
        col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print("  " + fmt.format(*headers))
    print("  " + "  ".join("-" * w for w in col_widths))
    for row in rows:
        print("  " + fmt.format(*[str(v) for v in row]))


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Compute Spearman rank correlation using stdlib only."""
    n = len(xs)
    if n < 3:
        return None

    def _rank(vals: list[float]) -> list[float]:
        indexed = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx = _rank(xs)
    ry = _rank(ys)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in ry))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def print_overall_counts(conn: sqlite3.Connection) -> None:
    _header("Overall Counts")
    for table, label in [
        ("binary_ratings", "Binary"),
        ("bestswap_ratings", "Best Swap"),
        ("goodswaps_ratings", "Good Swaps"),
        ("likert_ratings", "Likert"),
    ]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
        print(f"  {label:12s}: {count}")

    users = conn.execute(
        "SELECT COUNT(DISTINCT username) FROM sessions"
    ).fetchone()[0]
    print(f"  {'Unique users':12s}: {users}")


def print_per_user_stats(conn: sqlite3.Connection) -> None:
    _header("Per-User Stats")
    rows = conn.execute("""
        SELECT s.username,
               (SELECT COUNT(*) FROM binary_ratings b WHERE b.session_id = s.session_id),
               (SELECT COUNT(*) FROM bestswap_ratings bw WHERE bw.session_id = s.session_id),
               (SELECT COUNT(*) FROM goodswaps_ratings gs WHERE gs.session_id = s.session_id),
               (SELECT COUNT(*) FROM likert_ratings l WHERE l.session_id = s.session_id)
        FROM sessions s
        ORDER BY s.username
    """).fetchall()
    table_rows = []
    for username, bc, bsc, gsc, lc in rows:
        table_rows.append([username, bc, bsc, gsc, lc, bc + bsc + gsc + lc])
    _table(["User", "Binary", "BestSwap", "GoodSwaps", "Likert", "Total"], table_rows, [20, 8, 10, 10, 8, 8])


def print_binary_analysis(conn: sqlite3.Connection) -> None:
    _header("Binary Substitution Analysis")

    rows = conn.execute("SELECT response, candidate_rank, similarity_score FROM binary_ratings").fetchall()
    if not rows:
        print("  No binary ratings yet.")
        return

    total = len(rows)
    yes_count = sum(1 for r in rows if r[0] == "yes")
    print(f"  Overall acceptance rate: {yes_count}/{total} ({100*yes_count/total:.1f}%)")

    _subheader("Acceptance by candidate rank")
    by_rank: dict[int, list[str]] = defaultdict(list)
    for response, rank, _ in rows:
        by_rank[rank].append(response)
    table_rows = []
    for rank in sorted(by_rank):
        responses = by_rank[rank]
        y = sum(1 for r in responses if r == "yes")
        table_rows.append([rank, len(responses), y, f"{100*y/len(responses):.1f}%"])
    _table(["Rank", "Count", "Yes", "Rate"], table_rows, [6, 8, 6, 8])

    _subheader("Acceptance by similarity score bucket")
    buckets = [(0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    by_bucket: dict[str, list[str]] = {}
    for response, _, score in rows:
        for lo, hi in buckets:
            if lo <= score < hi:
                label = f"[{lo:.1f}, {hi:.1f})"
                by_bucket.setdefault(label, []).append(response)
                break
    table_rows = []
    for lo, hi in buckets:
        label = f"[{lo:.1f}, {hi:.1f})"
        responses = by_bucket.get(label, [])
        if responses:
            y = sum(1 for r in responses if r == "yes")
            table_rows.append([label, len(responses), y, f"{100*y/len(responses):.1f}%"])
    _table(["Score Bucket", "Count", "Yes", "Rate"], table_rows, [14, 8, 6, 8])


def print_bestswap_analysis(conn: sqlite3.Connection) -> None:
    _header("Best Swap Analysis")

    rows = conn.execute(
        "SELECT candidate_food_ids, chosen_food_id, candidate_ranks FROM bestswap_ratings"
    ).fetchall()
    if not rows:
        print("  No best swap ratings yet.")
        return

    chosen_ranks = []
    for cids_json, chosen_id, ranks_json in rows:
        cids = json.loads(cids_json)
        ranks = json.loads(ranks_json)
        # Find the rank of the chosen food
        for fid, rank in zip(cids, ranks):
            if fid == chosen_id:
                chosen_ranks.append(rank)
                break

    if not chosen_ranks:
        print("  Could not determine chosen ranks.")
        return

    total = len(chosen_ranks)
    print(f"  Total ratings: {total}")
    print(f"  Mean chosen rank: {statistics.mean(chosen_ranks):.2f}")
    if total > 1:
        print(f"  Median chosen rank: {statistics.median(chosen_ranks):.1f}")

    _subheader("Distribution of chosen ranks")
    rank_counts = Counter(chosen_ranks)
    table_rows = []
    for rank in sorted(rank_counts):
        count = rank_counts[rank]
        table_rows.append([rank, count, f"{100*count/total:.1f}%"])
    _table(["Rank", "Count", "Pct"], table_rows, [6, 8, 8])

    top1 = sum(1 for r in chosen_ranks if r == 1)
    print(f"\n  System top-1 chosen: {top1}/{total} ({100*top1/total:.1f}%)")


def print_goodswaps_analysis(conn: sqlite3.Connection) -> None:
    _header("Good Swaps Analysis")

    rows = conn.execute(
        "SELECT candidate_food_ids, chosen_food_ids, candidate_ranks FROM goodswaps_ratings"
    ).fetchall()
    if not rows:
        print("  No good swaps ratings yet.")
        return

    total = len(rows)
    all_chosen_counts = []
    chosen_by_rank: dict[int, int] = Counter()
    shown_by_rank: dict[int, int] = Counter()

    for cids_json, chosen_json, ranks_json in rows:
        cids = json.loads(cids_json)
        chosen = set(json.loads(chosen_json))
        ranks = json.loads(ranks_json)
        all_chosen_counts.append(len(chosen))
        for fid, rank in zip(cids, ranks):
            shown_by_rank[rank] += 1
            if fid in chosen:
                chosen_by_rank[rank] += 1

    print(f"  Total ratings: {total}")
    print(f"  Mean items selected per question: {statistics.mean(all_chosen_counts):.1f}")
    if total > 1:
        print(f"  Median items selected: {statistics.median(all_chosen_counts):.0f}")

    _subheader("Selection count distribution")
    count_dist = Counter(all_chosen_counts)
    table_rows = []
    for n in sorted(count_dist):
        c = count_dist[n]
        table_rows.append([n, c, f"{100*c/total:.1f}%"])
    _table(["Selected", "Count", "Pct"], table_rows, [10, 8, 8])

    _subheader("Selection rate by neighbor rank")
    table_rows = []
    for rank in sorted(shown_by_rank):
        shown = shown_by_rank[rank]
        chosen = chosen_by_rank.get(rank, 0)
        table_rows.append([rank, shown, chosen, f"{100*chosen/shown:.1f}%"])
    _table(["Rank", "Shown", "Selected", "Rate"], table_rows, [6, 8, 10, 8])


def print_likert_analysis(conn: sqlite3.Connection) -> None:
    _header("Likert Similarity Rating Analysis")

    rows = conn.execute(
        "SELECT rating, similarity_bin, similarity_score FROM likert_ratings"
    ).fetchall()
    if not rows:
        print("  No likert ratings yet.")
        return

    total = len(rows)
    ratings = [r[0] for r in rows]
    print(f"  Total ratings: {total}")
    print(f"  Mean rating: {statistics.mean(ratings):.2f}")
    if total > 1:
        print(f"  Std dev: {statistics.stdev(ratings):.2f}")

    _subheader("Rating distribution")
    rating_counts = Counter(ratings)
    table_rows = []
    for r in range(1, 6):
        count = rating_counts.get(r, 0)
        table_rows.append([r, count, f"{100*count/total:.1f}%"])
    _table(["Rating", "Count", "Pct"], table_rows, [8, 8, 8])

    _subheader("Mean human rating by similarity bin")
    by_bin: dict[int, list[int]] = defaultdict(list)
    for rating, sim_bin, _ in rows:
        by_bin[sim_bin].append(rating)
    bin_labels = {1: "Very different", 2: "Different group", 3: "Rank 6-10", 4: "Rank 3-5", 5: "Rank 1-2"}
    table_rows = []
    for b in range(1, 6):
        vals = by_bin.get(b, [])
        if vals:
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            table_rows.append([b, bin_labels.get(b, ""), len(vals), f"{mean:.2f}", f"{std:.2f}"])
    _table(["Bin", "Description", "Count", "Mean", "Std"], table_rows, [4, 18, 8, 8, 8])

    _subheader("Correlation: similarity bin vs human rating")
    bins = [r[1] for r in rows]
    rho = _spearman([float(b) for b in bins], [float(r) for r in ratings])
    if rho is not None:
        print(f"  Spearman rho (bin vs rating): {rho:.3f}  (n={total})")
    else:
        print("  Not enough data for correlation.")

    # Correlation with system score (where available)
    scored = [(r[2], r[0]) for r in rows if r[2] is not None]
    if len(scored) >= 3:
        _subheader("Correlation: system score vs human rating")
        scores, score_ratings = zip(*scored)
        rho2 = _spearman(list(scores), [float(r) for r in score_ratings])
        if rho2 is not None:
            print(f"  Spearman rho (system score vs rating): {rho2:.3f}  (n={len(scored)})")


def print_inter_rater_agreement(conn: sqlite3.Connection) -> None:
    _header("Inter-Rater Agreement")

    # Binary: check if same (query, candidate) pair was rated by multiple sessions
    rows = conn.execute("""
        SELECT b.query_food_id, b.candidate_food_id, s.username, b.response
        FROM binary_ratings b
        JOIN sessions s ON b.session_id = s.session_id
    """).fetchall()

    pair_ratings: dict[tuple, dict[str, str]] = defaultdict(dict)
    for qid, cid, user, response in rows:
        pair_ratings[(qid, cid)][user] = response

    overlap_pairs = {k: v for k, v in pair_ratings.items() if len(v) >= 2}
    if overlap_pairs:
        agreements = 0
        total_comparisons = 0
        for pair, user_responses in overlap_pairs.items():
            users = list(user_responses.keys())
            for i in range(len(users)):
                for j in range(i + 1, len(users)):
                    total_comparisons += 1
                    if user_responses[users[i]] == user_responses[users[j]]:
                        agreements += 1
        if total_comparisons > 0:
            print(
                f"  Binary: {len(overlap_pairs)} overlapping pairs, "
                f"{agreements}/{total_comparisons} pairwise agreements "
                f"({100 * agreements / total_comparisons:.1f}%)"
            )
    else:
        print("  Binary: no overlapping pairs between raters.")

    # Likert: check overlapping pairs
    rows = conn.execute("""
        SELECT l.food_id_a, l.food_id_b, s.username, l.rating
        FROM likert_ratings l
        JOIN sessions s ON l.session_id = s.session_id
    """).fetchall()

    pair_ratings_l: dict[tuple, dict[str, int]] = defaultdict(dict)
    for a, b, user, rating in rows:
        key = (min(a, b), max(a, b))
        pair_ratings_l[key][user] = rating

    overlap_pairs_l = {k: v for k, v in pair_ratings_l.items() if len(v) >= 2}
    if overlap_pairs_l:
        diffs = []
        for pair, user_ratings in overlap_pairs_l.items():
            users = list(user_ratings.keys())
            for i in range(len(users)):
                for j in range(i + 1, len(users)):
                    diffs.append(abs(user_ratings[users[i]] - user_ratings[users[j]]))
        print(f"  Likert: {len(overlap_pairs_l)} overlapping pairs, "
              f"mean absolute difference: {statistics.mean(diffs):.2f}")
    else:
        print("  Likert: no overlapping pairs between raters.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze manual evaluation results")
    parser.add_argument("--db", default="data/eval/eval.db", help="Path to eval database")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Run the evaluation web interface first to create the database.")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = None

    print_overall_counts(conn)
    print_per_user_stats(conn)
    print_binary_analysis(conn)
    print_bestswap_analysis(conn)
    print_goodswaps_analysis(conn)
    print_likert_analysis(conn)
    print_inter_rater_agreement(conn)

    conn.close()
    print()


if __name__ == "__main__":
    main()
