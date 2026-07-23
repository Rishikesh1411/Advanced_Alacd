#!/usr/bin/env python3
"""
Run baseline without context, baseline with context, CAD, DOLA, LACD, and
ALACD separately on each dataset and save full comparison results.

Written code by Rishikesh.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from data_utils import load_data_and_create_prompts
from model import Base_Model
from research_validation import (
    add_error_categories,
    compare_paired_runs,
    computational_cost_report,
    layer_distribution,
    sample_for_human_review,
    save_json_report,
)
from utils import (
    clean_generated_answer,
    clear_cuda_cache,
    compute_bertscore,
    compute_f1,
    compute_factscore_proxy,
    compute_rouge_scores,
    exact_or_contained_match,
    limit_answer_tokens,
    normalize_answer,
    normalize_gold_answers,
    remove_stop_words,
    set_seed,
)


QA_DATASETS = [
    "hotpot_qa",
    "squad",
    "squad_v2",
    "strategyqa",
    "boolq",
    "triviaqa",
    "natural_questions",
    "openbookqa",
    "arc_challenge",
    "commonsense_qa",
    "truthfulqa",
    "fever",
    "halueval",
]

SUMMARIZATION_DATASETS = ["cnn_dailymail", "xsum", "samsum"]

DEFAULT_METHODS = ["BASELINE_NO_CONTEXT", "BASELINE_WITH_CONTEXT", "CAD", "DOLA", "LACD", "ALACD"]

# Dataset/metric plan from the supplied hallucination-mitigation dataset PDF.
# Written code by Rishikesh: keep the paper-facing metric choice explicit so
# every result folder says which metric should decide the best method.
DATASET_METRIC_CATALOG = {
    "hotpot_qa": {
        "display_name": "HotPotQA",
        "purpose": "Multi-hop reasoning with supporting facts",
        "task_type": "qa",
        "primary_metric": "F1",
        "metrics": ["EM", "Partial Match", "F1", "Precision", "Recall"],
    },
    "squad": {
        "display_name": "SQuAD v1.1",
        "purpose": "Reading comprehension and question answering",
        "task_type": "qa",
        "primary_metric": "F1",
        "metrics": ["EM", "Partial Match", "F1", "Precision", "Recall"],
    },
    "squad_v2": {
        "display_name": "SQuAD v2.0",
        "purpose": "Answerable/unanswerable reading comprehension",
        "task_type": "qa",
        "primary_metric": "F1",
        "metrics": ["EM", "Partial Match", "F1", "Precision", "Recall"],
    },
    "truthfulqa": {
        "display_name": "TruthfulQA",
        "purpose": "Truthfulness and misconception avoidance",
        "task_type": "qa",
        "primary_metric": "F1",
        "metrics": ["EM", "Partial Match", "F1", "Precision", "Recall", "MC1/MC2/MC3 if using multiple-choice split"],
    },
    "fever": {
        "display_name": "FEVER",
        "purpose": "Claim verification and factual support",
        "task_type": "qa",
        "primary_metric": "F1",
        "metrics": ["EM", "Partial Match", "F1", "Precision", "Recall"],
    },
    "halueval": {
        "display_name": "HaluEval",
        "purpose": "Hallucination-specific factuality evaluation",
        "task_type": "qa",
        "primary_metric": "F1",
        "metrics": ["EM", "Partial Match", "F1", "Precision", "Recall"],
    },
    "cnn_dailymail": {
        "display_name": "CNN/DailyMail",
        "purpose": "News summarization",
        "task_type": "summarization",
        "primary_metric": "ROUGE-L F1",
        "metrics": ["ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1", "BERTScore F1", "FactScore Proxy"],
    },
    "xsum": {
        "display_name": "XSum",
        "purpose": "Abstractive summarization factuality",
        "task_type": "summarization",
        "primary_metric": "ROUGE-L F1",
        "metrics": ["ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1", "BERTScore F1", "FactScore Proxy"],
    },
    "samsum": {
        "display_name": "SAMSum",
        "purpose": "Dialogue summarization",
        "task_type": "summarization",
        "primary_metric": "ROUGE-L F1",
        "metrics": ["ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1", "BERTScore F1", "FactScore Proxy"],
    },
    "factscore": {
        "display_name": "FactScore",
        "purpose": "Factual correctness of generated text",
        "task_type": "metric_only",
        "primary_metric": "FactScore Proxy",
        "metrics": ["FactScore Proxy", "Human factuality"],
    },
}

METHOD_DEFAULTS = {
    "BASELINE_NO_CONTEXT": {"alpha": 0.0, "layer_alpha": 0.0, "start_layer": 0},
    "BASELINE_WITH_CONTEXT": {"alpha": 0.0, "layer_alpha": 0.0, "start_layer": 0},
    "CAD": {"alpha": 0.30, "layer_alpha": 0.50, "start_layer": 16},
    "DOLA": {"alpha": 0.30, "layer_alpha": 0.50, "start_layer": 16},
    "LACD": {"alpha": 0.30, "layer_alpha": 0.50, "start_layer": 16},
    "ALACD": {"alpha": 0.15, "layer_alpha": 0.50, "start_layer": 12},
}

METHOD_MODE_MAP = {
    # Baseline methods added by Rishikesh for direct comparison with CAD/DOLA/LACD/ALACD.
    "BASELINE_NO_CONTEXT": "final_layer_no_context",
    "BASELINE_WITH_CONTEXT": "final_layer_context",
    "CAD": "CAD",
    "DOLA": "DOLA",
    "LACD": "LACD",
    "ALACD": "ALACD",
}


@dataclass
class MethodResult:
    dataset: str
    task_type: str
    method: str
    metrics: Dict[str, float]
    details: List[dict]


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def model_layer_count(llm: Base_Model) -> int:
    config_layers = getattr(getattr(llm, "model", None), "config", None)
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(config_layers, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    # Problem 13 fix - modified by rishikesh.
    # OLD CODE:
    # adapter_layers = getattr(getattr(llm, "adapter", None), "num_layers", None)
    # if isinstance(adapter_layers, int) and adapter_layers > 0:
    #     return adapter_layers
    #
    # NEW CODE:
    # ModelAdapter exposes get_num_layers(), not a num_layers attribute.
    adapter = getattr(llm, "adapter", None)
    if adapter is not None and hasattr(adapter, "get_num_layers"):
        adapter_layers = adapter.get_num_layers()
        if isinstance(adapter_layers, int) and adapter_layers > 0:
            return adapter_layers
    return 32


def adjusted_start_layer(method: str, requested_start_layer: int, total_layers: int) -> int:
    if method in {"BASELINE_NO_CONTEXT", "BASELINE_WITH_CONTEXT"}:
        return 0
    if method == "CAD":
        return requested_start_layer
    if total_layers <= 4:
        return 0
    return max(0, min(requested_start_layer, total_layers - 3))


def build_dataset_args(dataset: str, task_type: str, max_examples: int, strategyqa_path: str = None):
    return SimpleNamespace(
        dataset=dataset,
        task_type=task_type,
        strategyqa_path=strategyqa_path,
        max_examples=max_examples,
    )


def load_dataset_examples(dataset: str, task_type: str, max_examples: int, strategyqa_path: str = None):
    data_args = build_dataset_args(dataset, task_type, max_examples, strategyqa_path)
    prompts_with_context, prompts_without_context, references = load_data_and_create_prompts(data_args)
    return (
        prompts_with_context[:max_examples],
        prompts_without_context[:max_examples],
        references[:max_examples],
    )


def generation_trace(llm: Base_Model) -> Tuple[List[int], float]:
    trace = getattr(llm, "last_generation_trace", []) or []
    selected_layers = [
        item.get("selected_layer")
        for item in trace
        if isinstance(item, dict) and item.get("selected_layer") is not None
    ]
    jsd_values = [
        float(item.get("max_jsd", 0.0))
        for item in trace
        if isinstance(item, dict)
    ]
    mean_jsd = sum(jsd_values) / max(len(jsd_values), 1)
    return selected_layers, mean_jsd


def evaluate_qa_method(
    llm: Base_Model,
    dataset: str,
    method: str,
    prompts_with_context: List[str],
    prompts_without_context: List[str],
    references: List[str],
    stop_word_list: List[str],
    params: Dict[str, float],
    max_new_tokens: int,
) -> MethodResult:
    rows = []
    totals = {"EM": 0.0, "Partial Match": 0.0, "F1": 0.0, "Precision": 0.0, "Recall": 0.0}

    total_count = len(prompts_with_context)
    with torch.inference_mode():
        for index, (prompt_ctx, prompt_no_ctx, reference) in enumerate(
            tqdm(
                zip(prompts_with_context, prompts_without_context, references),
                total=total_count,
                desc=f"{dataset}:{method}",
            )
        ):
            start = time.perf_counter()
            error = ""
            raw_output = ""
            try:
                # Baseline comparison fix - modified by rishikesh.
                # OLD CODE:
                # mode=method,
                #
                # NEW CODE:
                # Map readable comparison labels to the model's internal
                # baseline modes, then keep CAD/DOLA/LACD/ALACD unchanged.
                model_mode = METHOD_MODE_MAP.get(method, method)
                raw_output = llm.generate(
                    input_text=prompt_ctx,
                    input_text2=prompt_no_ctx,
                    mode=model_mode,
                    alpha=params["alpha"],
                    layer_alpha=params["layer_alpha"],
                    start_layer=params["start_layer"],
                    max_new_tokens=max_new_tokens,
                )
            except Exception as exc:
                error = str(exc)

            elapsed = time.perf_counter() - start
            selected_layers, mean_jsd = generation_trace(llm)
            cleaned = limit_answer_tokens(clean_generated_answer(raw_output), max_tokens=8)
            prediction = remove_stop_words(normalize_answer(cleaned), stop_word_list)
            # Problem 4 fix - modified by rishikesh.
            # OLD CODE:
            # gold = normalize_answer(reference)
            # em = 1 if prediction == gold else 0
            # partial = 1 if prediction and prediction in gold else 0
            # f1, precision, recall = compute_f1(gold, prediction)
            #
            # NEW CODE:
            # Compare predictions against every accepted gold alias. This keeps
            # SQuAD/SQuAD-v2 and TriviaQA from being under-scored or crashing
            # when the answer field is a list/dict.
            gold_aliases = normalize_gold_answers(reference)
            gold = gold_aliases[0]
            exact_match, partial_match, gold_aliases = exact_or_contained_match(prediction, gold_aliases)
            em = 1 if exact_match else 0
            partial = 1 if partial_match and not exact_match else 0
            f1, precision, recall = max(
                (compute_f1(alias, prediction) for alias in gold_aliases),
                key=lambda item: item[0],
            )

            generated_token_count = max(1, len(str(raw_output).split()))
            memory_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
            row = {
                "Dataset": dataset,
                "Task Type": "qa",
                "Method": method,
                "Index": index,
                "Question": prompt_ctx.split("Q:")[-1].split("A:")[0].strip(),
                "True Answer": gold,
                "Gold Aliases": " | ".join(gold_aliases),
                "Predicted Answer": prediction,
                "Raw Generated Text": raw_output,
                "EM": em,
                "Partial Match": partial,
                "F1": f1,
                "Precision": precision,
                "Recall": recall,
                "Selected Layers": selected_layers,
                "Mean JSD": mean_jsd,
                "tokens_per_sec": generated_token_count / max(elapsed, 1e-8),
                "inference_time_sec": elapsed,
                "latency_ms": elapsed * 1000.0,
                "memory_gb": memory_gb,
                "gpu_hours": elapsed / 3600.0 if torch.cuda.is_available() else 0.0,
                "estimated_cost_usd": 0.0,
                "error": error,
            }
            rows.append(row)

            for key in totals:
                totals[key] += float(row[key])
            clear_cuda_cache()

    rows = add_error_categories(rows)
    metrics = {key: value / max(total_count, 1) for key, value in totals.items()}
    metrics.update(
        {
            "examples": float(total_count),
            "mean_tokens_per_sec": sum(row["tokens_per_sec"] for row in rows) / max(total_count, 1),
            "mean_latency_ms": sum(row["latency_ms"] for row in rows) / max(total_count, 1),
            "mean_memory_gb": max((row["memory_gb"] for row in rows), default=0.0),
            "mean_jsd": sum(row["Mean JSD"] for row in rows) / max(total_count, 1),
            "errors": float(sum(1 for row in rows if row["error"])),
        }
    )
    return MethodResult(dataset, "qa", method, metrics, rows)


def evaluate_summarization_method(
    llm: Base_Model,
    dataset: str,
    method: str,
    prompts_with_context: List[str],
    prompts_without_context: List[str],
    references: List[str],
    params: Dict[str, float],
    max_new_tokens: int,
) -> MethodResult:
    rows = []
    totals = {
        "ROUGE-1 F1": 0.0,
        "ROUGE-1 Precision": 0.0,
        "ROUGE-1 Recall": 0.0,
        "ROUGE-2 F1": 0.0,
        "ROUGE-2 Precision": 0.0,
        "ROUGE-2 Recall": 0.0,
        "ROUGE-L F1": 0.0,
        "ROUGE-L Precision": 0.0,
        "ROUGE-L Recall": 0.0,
        "BERTScore F1": 0.0,
        "FactScore Proxy": 0.0,
    }

    total_count = len(prompts_with_context)
    with torch.inference_mode():
        for index, (prompt_ctx, prompt_no_ctx, reference) in enumerate(
            tqdm(
                zip(prompts_with_context, prompts_without_context, references),
                total=total_count,
                desc=f"{dataset}:{method}",
            )
        ):
            start = time.perf_counter()
            error = ""
            generated = ""
            try:
                # Baseline comparison fix - modified by rishikesh.
                # OLD CODE:
                # mode=method,
                #
                # NEW CODE:
                model_mode = METHOD_MODE_MAP.get(method, method)
                generated = llm.generate(
                    input_text=prompt_ctx,
                    input_text2=prompt_no_ctx,
                    mode=model_mode,
                    alpha=params["alpha"],
                    layer_alpha=params["layer_alpha"],
                    start_layer=params["start_layer"],
                    max_new_tokens=max_new_tokens,
                ).strip()
            except Exception as exc:
                error = str(exc)

            elapsed = time.perf_counter() - start
            selected_layers, mean_jsd = generation_trace(llm)

            if generated and reference:
                rouge = compute_rouge_scores(reference, generated)
                bert = compute_bertscore(reference, generated)
                fact_score = compute_factscore_proxy(reference, generated)
            else:
                rouge = {
                    "rouge1": {"f": 0.0, "p": 0.0, "r": 0.0},
                    "rouge2": {"f": 0.0, "p": 0.0, "r": 0.0},
                    "rougel": {"f": 0.0, "p": 0.0, "r": 0.0},
                }
                bert = {"f1": 0.0}
                fact_score = 0.0

            generated_token_count = max(1, len(generated.split()))
            memory_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
            row = {
                "Dataset": dataset,
                "Task Type": "summarization",
                "Method": method,
                "Index": index,
                "Reference Summary": reference,
                "Generated Summary": generated,
                "ROUGE-1 F1": rouge["rouge1"]["f"],
                "ROUGE-1 Precision": rouge["rouge1"]["p"],
                "ROUGE-1 Recall": rouge["rouge1"]["r"],
                "ROUGE-2 F1": rouge["rouge2"]["f"],
                "ROUGE-2 Precision": rouge["rouge2"]["p"],
                "ROUGE-2 Recall": rouge["rouge2"]["r"],
                "ROUGE-L F1": rouge["rougel"]["f"],
                "ROUGE-L Precision": rouge["rougel"]["p"],
                "ROUGE-L Recall": rouge["rougel"]["r"],
                "BERTScore F1": bert["f1"],
                "FactScore Proxy": fact_score,
                "Selected Layers": selected_layers,
                "Mean JSD": mean_jsd,
                "tokens_per_sec": generated_token_count / max(elapsed, 1e-8),
                "inference_time_sec": elapsed,
                "latency_ms": elapsed * 1000.0,
                "memory_gb": memory_gb,
                "gpu_hours": elapsed / 3600.0 if torch.cuda.is_available() else 0.0,
                "estimated_cost_usd": 0.0,
                "error": error,
            }
            rows.append(row)

            for key in totals:
                totals[key] += float(row[key])
            clear_cuda_cache()

    metrics = {key: value / max(total_count, 1) for key, value in totals.items()}
    metrics.update(
        {
            "examples": float(total_count),
            "mean_tokens_per_sec": sum(row["tokens_per_sec"] for row in rows) / max(total_count, 1),
            "mean_latency_ms": sum(row["latency_ms"] for row in rows) / max(total_count, 1),
            "mean_memory_gb": max((row["memory_gb"] for row in rows), default=0.0),
            "mean_jsd": sum(row["Mean JSD"] for row in rows) / max(total_count, 1),
            "errors": float(sum(1 for row in rows if row["error"])),
        }
    )
    return MethodResult(dataset, "summarization", method, metrics, rows)


def save_dataset_outputs(
    dataset_dir: Path,
    dataset: str,
    task_type: str,
    method_results: List[MethodResult],
    human_eval_sample: bool,
    human_eval_n: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset_dir.mkdir(parents=True, exist_ok=True)

    detail_frames = []
    summary_rows = []
    for result in method_results:
        details_df = pd.DataFrame(result.details)
        details_path = dataset_dir / f"{safe_name(result.method)}_details.csv"
        details_df.to_csv(details_path, index=False)
        detail_frames.append(details_df)

        summary_row = {"Dataset": dataset, "Task Type": task_type, "Method": result.method}
        summary_row.update(result.metrics)
        summary_rows.append(summary_row)

        layer_distribution(result.details, str(dataset_dir / f"{safe_name(result.method)}_layer_distribution.csv"))
        computational_cost_report(result.details, str(dataset_dir / f"{safe_name(result.method)}_cost_report.csv"))
        if human_eval_sample:
            sample_for_human_review(
                result.details,
                str(dataset_dir / f"{safe_name(result.method)}_human_review.csv"),
                n=human_eval_n,
                seed=seed,
            )

    all_details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    method_summary = pd.DataFrame(summary_rows)
    all_details.to_csv(dataset_dir / f"{safe_name(dataset)}_all_method_details.csv", index=False)
    method_summary.to_csv(dataset_dir / f"{safe_name(dataset)}_method_summary.csv", index=False)

    # Problem 5 / 11 fix - modified by rishikesh.
    # OLD CODE:
    # best_metric = "F1" if task_type == "qa" else "ROUGE-L F1"
    #
    # NEW CODE:
    # Use the dataset-specific primary metric from the PDF catalog when present.
    best_metric = DATASET_METRIC_CATALOG.get(dataset, {}).get(
        "primary_metric",
        "F1" if task_type == "qa" else "ROUGE-L F1",
    )
    if best_metric in method_summary.columns:
        method_summary.sort_values(best_metric, ascending=False).to_csv(
            dataset_dir / f"{safe_name(dataset)}_ranked_by_{safe_name(best_metric)}.csv",
            index=False,
        )

    return all_details, method_summary


def paired_reports(dataset_dir: Path, task_type: str, all_details: pd.DataFrame, baseline_method: str = "LACD"):
    if all_details.empty or baseline_method not in set(all_details["Method"]):
        return {}

    metric = "F1" if task_type == "qa" else "ROUGE-L F1"
    reports = {}
    baseline_rows = all_details[all_details["Method"] == baseline_method].sort_values("Index").to_dict("records")
    for method in sorted(set(all_details["Method"])):
        if method == baseline_method:
            continue
        method_rows = all_details[all_details["Method"] == method].sort_values("Index").to_dict("records")
        if len(method_rows) != len(baseline_rows):
            continue
        report = compare_paired_runs(baseline_rows, method_rows, metric=metric)
        reports[f"{method}_vs_{baseline_method}"] = report
        save_json_report(report, str(dataset_dir / f"{safe_name(method)}_vs_{safe_name(baseline_method)}_paired_stats.json"))
    return reports


def task_type_for_dataset(dataset: str) -> str:
    catalog_task = DATASET_METRIC_CATALOG.get(dataset, {}).get("task_type")
    if catalog_task in {"qa", "summarization"}:
        return catalog_task
    return "summarization" if dataset in SUMMARIZATION_DATASETS else "qa"


def selected_datasets(value: str) -> List[str]:
    if value == "all":
        return QA_DATASETS + SUMMARIZATION_DATASETS
    if value == "qa":
        return QA_DATASETS
    if value == "summarization":
        return SUMMARIZATION_DATASETS
    return [item.strip() for item in value.split(",") if item.strip()]


def save_dataset_metric_catalog(results_root: Path) -> Path:
    rows = []
    for dataset, meta in DATASET_METRIC_CATALOG.items():
        rows.append(
            {
                "dataset": dataset,
                "display_name": meta["display_name"],
                "task_type": meta["task_type"],
                "purpose": meta["purpose"],
                "primary_metric": meta["primary_metric"],
                "all_metrics": " | ".join(meta["metrics"]),
                "runner_support": dataset in QA_DATASETS or dataset in SUMMARIZATION_DATASETS,
            }
        )
    path = results_root / "dataset_metric_catalog.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def save_best_method_summary(combined_summary: pd.DataFrame, results_root: Path) -> Path:
    # Problem 4 / 5 / 11 fix - modified by rishikesh.
    # OLD CODE:
    # # No single file showed which method was best for each dataset.
    #
    # NEW CODE:
    # Create a reviewer-friendly table: dataset, metric used, winning method,
    # winning score, and all methods sorted by that dataset's primary metric.
    rows = []
    for dataset, group in combined_summary.groupby("Dataset", dropna=False):
        task_type = str(group["Task Type"].iloc[0])
        metric = DATASET_METRIC_CATALOG.get(dataset, {}).get(
            "primary_metric",
            "F1" if task_type == "qa" else "ROUGE-L F1",
        )
        if metric not in group.columns:
            continue
        ranked = group.sort_values(metric, ascending=False)
        best = ranked.iloc[0]
        rows.append(
            {
                "Dataset": dataset,
                "Task Type": task_type,
                "Primary Metric": metric,
                "Best Method": best["Method"],
                "Best Score": best[metric],
                "Method Ranking": " > ".join(ranked["Method"].astype(str).tolist()),
                "All Metrics From PDF": " | ".join(DATASET_METRIC_CATALOG.get(dataset, {}).get("metrics", [])),
            }
        )
    path = results_root / "best_method_by_dataset.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def load_model(model_name: str, device: str, num_gpus: str, max_gpu_memory: int, precision: str) -> Base_Model:
    llm = Base_Model(
        model_name=model_name,
        device=device,
        num_gpus=num_gpus,
        max_gpu_memory=max_gpu_memory,
        # Problem 9 fix - modified by rishikesh.
        # OLD CODE:
        # # precision was not passed through this runner.
        #
        # NEW CODE:
        # # Save and use explicit bf16/fp16/fp32 precision for reproducibility.
        precision=precision,
    )
    clear_cuda_cache()
    return llm


def run(args):
    set_seed(args.seed)
    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)
    catalog_path = save_dataset_metric_catalog(results_root)

    datasets = selected_datasets(args.datasets)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]

    print("=" * 100)
    print("BASELINE / CAD / DOLA / LACD / ALACD DATASET-BY-DATASET COMPARISON")
    print("Written code by Rishikesh")
    print("=" * 100)
    print(f"Model: {args.model_name}")
    print(f"Datasets: {datasets}")
    print(f"Methods: {methods}")
    print(f"Max examples per dataset: {args.max_examples}")
    print(f"Results directory: {results_root}")
    print(f"Dataset metric catalog: {catalog_path}")
    print("=" * 100)

    llm = load_model(args.model_name, args.device, args.num_gpus, args.max_gpu_memory, args.precision)
    stop_word_list = ["Q:", "Supporting information:", "\n", "\n\n##"]
    llm.set_stop_words(stop_word_list)

    total_layers = model_layer_count(llm)
    print(f"Detected model layers: {total_layers}")

    combined_details = []
    combined_summaries = []
    manifest = {
        "written_code_by": "Rishikesh",
        "model_name": args.model_name,
        "datasets": datasets,
        "methods": methods,
        "max_examples": args.max_examples,
        "max_new_tokens_qa": args.max_new_tokens_qa,
        "max_new_tokens_summarization": args.max_new_tokens_summarization,
        "precision": args.precision,
        "seed": args.seed,
        "runs": [],
    }

    for dataset in datasets:
        task_type = task_type_for_dataset(dataset)
        dataset_dir = results_root / safe_name(dataset)
        print("\n" + "=" * 100)
        print(f"DATASET: {dataset} | TASK: {task_type}")
        print("=" * 100)

        try:
            prompts_ctx, prompts_no_ctx, references = load_dataset_examples(
                dataset, task_type, args.max_examples, args.strategyqa_path
            )
        except Exception as exc:
            error_path = dataset_dir / "dataset_load_error.txt"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            error_path.write_text(str(exc), encoding="utf-8")
            manifest["runs"].append({"dataset": dataset, "task_type": task_type, "status": "load_failed", "error": str(exc)})
            print(f"Dataset load failed for {dataset}: {exc}")
            continue

        method_results = []
        for method in methods:
            params = dict(METHOD_DEFAULTS.get(method, METHOD_DEFAULTS["ALACD"]))
            params["start_layer"] = adjusted_start_layer(method, params["start_layer"], total_layers)
            max_new_tokens = args.max_new_tokens_summarization if task_type == "summarization" else args.max_new_tokens_qa
            print(f"\nRunning {method} on {dataset} with params {params}")
            try:
                if task_type == "summarization":
                    result = evaluate_summarization_method(
                        llm,
                        dataset,
                        method,
                        prompts_ctx,
                        prompts_no_ctx,
                        references,
                        params,
                        max_new_tokens,
                    )
                else:
                    result = evaluate_qa_method(
                        llm,
                        dataset,
                        method,
                        prompts_ctx,
                        prompts_no_ctx,
                        references,
                        stop_word_list,
                        params,
                        max_new_tokens,
                    )
                method_results.append(result)
                manifest["runs"].append(
                    {
                        "dataset": dataset,
                        "task_type": task_type,
                        "method": method,
                        "status": "ok",
                        "metrics": result.metrics,
                    }
                )
            except Exception as exc:
                manifest["runs"].append(
                    {"dataset": dataset, "task_type": task_type, "method": method, "status": "failed", "error": str(exc)}
                )
                print(f"Method failed: dataset={dataset} method={method} error={exc}")
            clear_cuda_cache()

        if method_results:
            all_details, method_summary = save_dataset_outputs(
                dataset_dir,
                dataset,
                task_type,
                method_results,
                args.human_eval_sample,
                args.human_eval_n,
                args.seed,
            )
            paired_reports(dataset_dir, task_type, all_details, baseline_method=args.stats_baseline)
            save_dataset_metric_catalog(dataset_dir)
            combined_details.append(all_details)
            combined_summaries.append(method_summary)

            print("\nDataset summary:")
            print(method_summary.to_string(index=False))

    if combined_details:
        pd.concat(combined_details, ignore_index=True).to_csv(results_root / "all_datasets_all_method_details.csv", index=False)
    if combined_summaries:
        combined_summary = pd.concat(combined_summaries, ignore_index=True)
        combined_summary.to_csv(results_root / "all_datasets_method_summary.csv", index=False)
        best_path = save_best_method_summary(combined_summary, results_root)
        print("\n" + "=" * 100)
        print("FINAL COMBINED SUMMARY")
        print("=" * 100)
        print(combined_summary.to_string(index=False))
        print(f"\nBest method by dataset saved to: {best_path}")

    manifest_path = results_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved run manifest: {manifest_path}")
    print(f"Saved all results under: {results_root}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run baselines, CAD, DOLA, LACD, and ALACD separately on each dataset with full metric comparison."
    )
    parser.add_argument("--model-name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--datasets", type=str, default="all", help="'all', 'qa', 'summarization', or comma-separated names")
    parser.add_argument("--methods", type=str, default="BASELINE_NO_CONTEXT,BASELINE_WITH_CONTEXT,CAD,DOLA,LACD,ALACD")
    parser.add_argument("--max-examples", type=int, default=25)
    parser.add_argument("--max-new-tokens-qa", type=int, default=25)
    parser.add_argument("--max-new-tokens-summarization", type=int, default=120)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max-gpu-memory", type=int, default=24)
    parser.add_argument("--precision", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--results-dir", type=str, default="results/all_methods_comparison")
    parser.add_argument("--strategyqa-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--human-eval-sample", action="store_true")
    parser.add_argument("--human-eval-n", type=int, default=75)
    parser.add_argument("--stats-baseline", type=str, default="LACD")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
