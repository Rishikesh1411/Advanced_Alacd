# ============================================================
# ALACD 14 PPT PROBLEM FIXES - Written code by Rishikesh
# This generated file contains the concrete implementation used by the notebook.
# See the ledger cell for the problem-number mapping.
# ============================================================

"""
Research validation and reporting helpers for ALACD.

Written by Rishikesh Raj: this module keeps experimental validation separate

Problem 3 - Statistical Significance Testing: bootstrap CI, paired t-test, McNemar.
Problem 6 - Human Evaluation: reviewer sampling sheet.
Problem 8 - High JSD Evidence: JSD vs metric correlation and theory helpers.
Problem 9 - Reproducibility: hyperparameter table writer.
Problem 12 - Missing Error Analysis: entity/date/number/reasoning/over-correction categories.
Problem 13 - Layer Distribution: selected-layer histogram CSV.
Problem 14 - Computational Cost Report: latency, tokens/sec, memory, GPU-hours, cost table.
from decoding so the paper numbers can be reproduced instead of hand-edited.
"""

import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class ReproducibilityConfig:
    beta: float
    lambda_value: float
    K: int
    LS: str
    actx: str
    alayer: str
    temp_min: float
    temp_max: float
    seed: int
    mode: str
    model_name: str
    dataset: str


def _as_float_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray([float(v) for v in values], dtype=np.float64)


def bootstrap_ci(values: Sequence[float], n_bootstrap: int = 2000, ci: float = 0.95, seed: int = 42) -> Dict[str, float]:
    """
    OLD CODE:
    # No confidence interval was reported.

    NEW CODE by Rishikesh:
    Bootstrap the mean and return lower/upper confidence bounds.
    """
    arr = _as_float_array(values)
    if arr.size == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boot[i] = sample.mean()
    alpha = (1.0 - ci) / 2.0
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.quantile(boot, alpha)),
        "ci_high": float(np.quantile(boot, 1.0 - alpha)),
    }


def paired_t_test(before: Sequence[float], after: Sequence[float]) -> Dict[str, float]:
    """
    OLD CODE:
    # Improvements were printed without a paired significance test.

    NEW CODE by Rishikesh:
    Paired t-test over per-example metric deltas.
    """
    x = _as_float_array(before)
    y = _as_float_array(after)
    n = min(x.size, y.size)
    if n < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "mean_delta": 0.0}
    delta = y[:n] - x[:n]
    mean_delta = float(delta.mean())
    std_delta = float(delta.std(ddof=1))
    if std_delta == 0.0:
        return {"t_stat": 0.0, "p_value": 1.0 if mean_delta == 0.0 else 0.0, "mean_delta": mean_delta}
    t_stat = mean_delta / (std_delta / math.sqrt(n))
    try:
        from scipy import stats
        p_value = float(stats.ttest_rel(y[:n], x[:n], nan_policy="omit").pvalue)
    except Exception:
        # Normal approximation fallback when scipy is unavailable.
        p_value = float(math.erfc(abs(t_stat) / math.sqrt(2.0)))
    return {"t_stat": float(t_stat), "p_value": p_value, "mean_delta": mean_delta}


def mcnemar_test(before_correct: Sequence[int], after_correct: Sequence[int]) -> Dict[str, float]:
    """
    OLD CODE:
    # EM improvements were compared by average only.

    NEW CODE by Rishikesh:
    McNemar test on paired exact-match correctness.
    """
    before = np.asarray(before_correct, dtype=bool)
    after = np.asarray(after_correct, dtype=bool)
    n = min(before.size, after.size)
    b = int(np.logical_and(before[:n], np.logical_not(after[:n])).sum())
    c = int(np.logical_and(np.logical_not(before[:n]), after[:n]).sum())
    if b + c == 0:
        return {"b_before_only": b, "c_after_only": c, "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
    try:
        from scipy.stats import chi2 as chi2_dist
        p_value = float(chi2_dist.sf(chi2, df=1))
    except Exception:
        p_value = float(math.erfc(math.sqrt(chi2 / 2.0)))
    return {"b_before_only": b, "c_after_only": c, "chi2": float(chi2), "p_value": p_value}


def compare_paired_runs(baseline_rows: List[dict], improved_rows: List[dict], metric: str = "F1") -> Dict[str, object]:
    baseline = [row.get(metric, 0.0) for row in baseline_rows]
    improved = [row.get(metric, 0.0) for row in improved_rows]
    deltas = [float(i) - float(b) for b, i in zip(baseline, improved)]
    return {
        "metric": metric,
        "baseline_ci": bootstrap_ci(baseline),
        "improved_ci": bootstrap_ci(improved),
        "delta_ci": bootstrap_ci(deltas),
        "paired_t_test": paired_t_test(baseline, improved),
        "mcnemar": mcnemar_test(
            [row.get("EM", 0) for row in baseline_rows],
            [row.get("EM", 0) for row in improved_rows],
        ),
    }


def sample_for_human_review(rows: List[dict], output_path: str, n: int = 75, seed: int = 42) -> str:
    """
    OLD CODE:
    # No human-evaluation review sheet was produced.

    NEW CODE by Rishikesh:
    Sample 50-100 outputs and create reviewer columns for factuality,
    faithfulness, fluency, and notes.
    """
    n = int(max(50, min(100, n)))
    rng = np.random.default_rng(seed)
    if len(rows) > n:
        idx = rng.choice(len(rows), size=n, replace=False)
        sampled = [rows[int(i)] for i in idx]
    else:
        sampled = list(rows)
    df = pd.DataFrame(sampled)
    for col in ["human_factuality_1_5", "human_faithfulness_1_5", "human_fluency_1_5", "reviewer_notes"]:
        if col not in df.columns:
            df[col] = ""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def categorize_error(question: str, reference: str, prediction: str) -> str:
    text = f"{question} {reference} {prediction}".lower()
    pred = str(prediction).lower()
    ref = str(reference).lower()
    if any(ch.isdigit() for ch in pred + ref):
        if any(token in text for token in ["year", "date", "born", "died", "when"]):
            return "date"
        return "number"
    if ref and pred and ref not in pred and pred not in ref:
        if any(token in text for token in ["who", "person", "company", "city", "country", "president", "minister"]):
            return "entity"
    if len(pred.split()) <= 2 and len(ref.split()) > 5:
        return "over-correction"
    if any(token in text for token in ["why", "how", "because", "therefore"]):
        return "reasoning"
    if "supporting information" in text or "context" in text:
        return "context misinterpretation"
    return "entity"


def add_error_categories(rows: List[dict]) -> List[dict]:
    enriched = []
    for row in rows:
        new_row = dict(row)
        new_row["error_category"] = categorize_error(
            str(row.get("Question", "")),
            str(row.get("True Answer", row.get("Reference Summary", ""))),
            str(row.get("Predicted Answer", row.get("Generated Summary", ""))),
        )
        enriched.append(new_row)
    return enriched


def layer_distribution(rows: List[dict], output_path: str) -> str:
    layers = []
    for row in rows:
        value = row.get("Selected Layers", row.get("selected_layers", ""))
        if isinstance(value, str):
            for part in value.replace("[", "").replace("]", "").split(","):
                part = part.strip()
                if part.isdigit():
                    layers.append(int(part))
        elif isinstance(value, Iterable):
            layers.extend(int(v) for v in value if v is not None)
    counts = Counter(layers)
    df = pd.DataFrame([{"layer": k, "count": v} for k, v in sorted(counts.items())])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def jsd_hallucination_correlation(rows: List[dict], metric: str = "F1") -> Dict[str, float]:
    jsd = []
    gains = []
    for row in rows:
        if "Mean JSD" in row and metric in row:
            jsd.append(float(row["Mean JSD"]))
            gains.append(float(row[metric]))
    if len(jsd) < 2:
        return {"pearson_r": 0.0, "spearman_r": 0.0, "n": len(jsd)}
    try:
        from scipy import stats
        pearson = float(stats.pearsonr(jsd, gains).statistic)
        spearman = float(stats.spearmanr(jsd, gains).statistic)
    except Exception:
        pearson = float(np.corrcoef(jsd, gains)[0, 1])
        spearman = pearson
    return {"pearson_r": pearson, "spearman_r": spearman, "n": len(jsd)}


def theoretical_analysis(max_jsd: float, p_final: float, p_mid: float, entropy: float) -> Dict[str, float]:
    alpha = 0.0
    temp = 1.0
    from utils import AdaptiveAlpha, TempFromEntropy
    alpha = AdaptiveAlpha(max_jsd, p_final, p_mid)
    temp = TempFromEntropy(entropy)
    return {
        "adaptive_alpha": alpha,
        "temperature": temp,
        "interpretation": (
            "Higher JSD increases contrast only when the mid-layer distribution "
            "is sufficiently confident; higher entropy raises temperature to "
            "avoid brittle token collapse."
        ),
    }


def reproducibility_table(configs: List[ReproducibilityConfig], output_path: str) -> str:
    df = pd.DataFrame([asdict(cfg) for cfg in configs])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def computational_cost_report(rows: List[dict], output_path: str) -> str:
    df = pd.DataFrame(rows)
    for col in ["tokens_per_sec", "inference_time_sec", "gpu_hours", "latency_ms", "memory_gb", "estimated_cost_usd"]:
        if col not in df.columns:
            df[col] = 0.0
    # Problem 14 fix - modified by rishikesh.
    # OLD CODE:
    # grouped = df.groupby("mode", dropna=False)[
    #     ["tokens_per_sec", "inference_time_sec", "gpu_hours", "latency_ms", "memory_gb", "estimated_cost_usd"]
    # ].mean().reset_index()
    #
    # NEW CODE:
    # The all-method runner stores the decoding method under "Method"; older
    # main.py rows may still use "mode". Support both so cost reports never
    # fail after a successful experiment.
    group_col = "Method" if "Method" in df.columns else "mode"
    if group_col not in df.columns:
        df[group_col] = "unknown"
    grouped = df.groupby(group_col, dropna=False)[
        ["tokens_per_sec", "inference_time_sec", "gpu_hours", "latency_ms", "memory_gb", "estimated_cost_usd"]
    ].mean().reset_index()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    grouped.to_csv(output_path, index=False)
    return output_path


class GenerationTimer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        return False


def save_json_report(report: dict, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return output_path
