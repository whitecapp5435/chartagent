from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


@dataclass(frozen=True)
class LabelCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0


def multilabel_counts(
    y_true: Sequence[Set[str]],
    y_pred: Sequence[Set[str]],
    labels: Sequence[str],
) -> Dict[str, LabelCounts]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    label_set = list(labels)
    counts: Dict[str, LabelCounts] = {l: LabelCounts() for l in label_set}

    tp: Dict[str, int] = {l: 0 for l in label_set}
    fp: Dict[str, int] = {l: 0 for l in label_set}
    fn: Dict[str, int] = {l: 0 for l in label_set}

    for t, p in zip(y_true, y_pred):
        for l in label_set:
            in_t = l in t
            in_p = l in p
            if in_t and in_p:
                tp[l] += 1
            elif (not in_t) and in_p:
                fp[l] += 1
            elif in_t and (not in_p):
                fn[l] += 1

    for l in label_set:
        counts[l] = LabelCounts(tp=tp[l], fp=fp[l], fn=fn[l])
    return counts


def multilabel_exact_match(y_true: Sequence[Set[str]], y_pred: Sequence[Set[str]]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if set(t) == set(p))
    return correct / float(len(y_true))


def multilabel_partial_match(y_true: Sequence[Set[str]], y_pred: Sequence[Set[str]]) -> float:
    """Example-level partial match: at least one correct label predicted.

    Returns fraction of examples where `y_true ∩ y_pred` is non-empty.
    If both sets are empty, counts as a match.
    """

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return 0.0
    ok = 0
    for t, p in zip(y_true, y_pred):
        tset = set(t)
        pset = set(p)
        if not tset and not pset:
            ok += 1
        elif tset.intersection(pset):
            ok += 1
    return ok / float(len(y_true))


def multilabel_jaccard_accuracy(y_true: Sequence[Set[str]], y_pred: Sequence[Set[str]]) -> float:
    """Example-level Jaccard similarity (often called 'accuracy' for multi-label)."""

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return 0.0
    total = 0.0
    for t, p in zip(y_true, y_pred):
        tset = set(t)
        pset = set(p)
        union = tset.union(pset)
        if not union:
            total += 1.0
        else:
            total += float(len(tset.intersection(pset))) / float(len(union))
    return total / float(len(y_true))


def multilabel_micro_f1(counts: Dict[str, LabelCounts]) -> Tuple[float, float, float]:
    tp = sum(c.tp for c in counts.values())
    fp = sum(c.fp for c in counts.values())
    fn = sum(c.fn for c in counts.values())
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * prec * rec, prec + rec)
    return prec, rec, f1


def multilabel_macro_f1(counts: Dict[str, LabelCounts]) -> Tuple[float, float, float]:
    if not counts:
        return 0.0, 0.0, 0.0
    per_prec: List[float] = []
    per_rec: List[float] = []
    per_f1: List[float] = []
    for c in counts.values():
        prec = _safe_div(c.tp, c.tp + c.fp)
        rec = _safe_div(c.tp, c.tp + c.fn)
        f1 = _safe_div(2.0 * prec * rec, prec + rec)
        per_prec.append(prec)
        per_rec.append(rec)
        per_f1.append(f1)
    macro_p = sum(per_prec) / float(len(per_prec))
    macro_r = sum(per_rec) / float(len(per_rec))
    macro_f1 = sum(per_f1) / float(len(per_f1))
    return macro_p, macro_r, macro_f1


def multilabel_score(
    y_true: Sequence[Set[str]],
    y_pred: Sequence[Set[str]],
    labels: Sequence[str],
) -> Dict[str, float]:
    counts = multilabel_counts(y_true, y_pred, labels)
    micro_p, micro_r, micro_f1 = multilabel_micro_f1(counts)
    macro_p, macro_r, macro_f1 = multilabel_macro_f1(counts)
    return {
        "n": float(len(y_true)),
        "accuracy": multilabel_jaccard_accuracy(y_true, y_pred),
        "exact_match": multilabel_exact_match(y_true, y_pred),
        "partial_match": multilabel_partial_match(y_true, y_pred),
        "precision": micro_p,
        "recall": micro_r,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
    }
