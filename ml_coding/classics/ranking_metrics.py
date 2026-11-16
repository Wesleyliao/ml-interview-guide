import numpy as np


def roc_auc(y_true, y_score):
    # y_true: (n,) in {0, 1}, y_score: (n,) predicted scores.
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    # 1. Sort from highest score to lowest score.
    order = np.argsort(-y_score, kind="stable")
    y_sorted, s_sorted = y_true[order], y_score[order]

    # 2. Walk down distinct score groups and sum trapezoid slices under the ROC curve.
    group_ends = np.flatnonzero(np.r_[s_sorted[:-1] != s_sorted[1:], True])
    tp = area = 0.0
    start = 0
    for end in group_ends:
        pos_step = y_sorted[start : end + 1].sum()
        neg_step = (end - start + 1) - pos_step
        area += (tp + tp + pos_step) * 0.5 * neg_step
        tp += pos_step
        start = end + 1

    return float(area / (n_pos * n_neg))


def ndcg_at_k(relevances, scores, k):
    # relevances: (n,) graded relevance, scores: (n,) model scores.
    rel = np.asarray(relevances, dtype=float)
    sc = np.asarray(scores, dtype=float)
    k = min(k, len(rel))
    discounts = np.log2(np.arange(2, k + 2))

    # 1. DCG@k of model ranking vs. IDCG@k of ideal ranking.
    top_rel = rel[np.argsort(-sc, kind="stable")[:k]]
    ideal_rel = np.sort(rel)[::-1][:k]
    dcg = np.sum((2.0 ** top_rel - 1.0) / discounts)
    idcg = np.sum((2.0 ** ideal_rel - 1.0) / discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0
