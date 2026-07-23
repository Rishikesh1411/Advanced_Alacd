# evaluation.py

import os
import torch
import pandas as pd
from tqdm import tqdm

from utils import (
    normalize_answer,
    normalize_gold_answers,
    exact_or_contained_match,
    remove_stop_words,
    compute_f1,
    clean_generated_answer,
    limit_answer_tokens,
)

# NEW CODE by Rishikesh: task-specific quality measures (RAUQ/PRR, Accuracy,
# COMET, AlignScore). See quality_metrics.py for the honesty notes on each.
from quality_metrics import (
    compute_accuracy_qa,
    compute_comet_scores,
    compute_alignscore_scores,
    compute_sequence_uncertainty,
    compute_prr,
)


def evaluate_model(llm, prompts_with_context, prompts_without_context, answers, stop_word_list, args):
    """
    Evaluates the model on given prompts (with and without context),
    then calculates EM (Exact Match), partial match, F1, Precision, and Recall.
    Also returns a quality_metrics dict with Accuracy, COMET, AlignScore, and
    RAUQ/PRR (each opt-in via --use-comet / --use-alignscore / --use-rauq,
    except Accuracy which is free since it equals EM for QA).
    """
    em_score = 0
    partial_match_score = 0
    f1_score_total = 0
    precision_total = 0
    recall_total = 0
    total_count = len(prompts_with_context)

    use_rauq = bool(getattr(args, "use_rauq", False))
    use_comet = bool(getattr(args, "use_comet", False))
    use_alignscore = bool(getattr(args, "use_alignscore", False))

    per_example_uncertainty = []
    per_example_correctness = []
    # For COMET/AlignScore we batch-score at the end rather than per-example,
    # so collect the raw text needed for each.
    comet_sources, comet_hyps, comet_refs = [], [], []
    alignscore_contexts, alignscore_claims = [], []

    incorrect_details = []

    with torch.no_grad():
        for index, (prompt_w_ctx, prompt_no_ctx, true_answer) in enumerate(
            tqdm(zip(prompts_with_context, prompts_without_context, answers), total=total_count)
        ):
            # Generate response from the model
            max_new_tokens = getattr(args, 'max_new_tokens', 25)
            raw_generated_response = llm.generate(
                input_text=prompt_w_ctx,
                input_text2=prompt_no_ctx,
                mode=args.mode,
                alpha=args.alpha,
                layer_alpha=args.layer_alpha,
                start_layer=args.start_layer,
                max_new_tokens=max_new_tokens
            )

            # Post-process generated text to better match gold span answers
            generated_response = clean_generated_answer(raw_generated_response)
            generated_response = limit_answer_tokens(generated_response, max_tokens=8)

            # Normalize gold/predicted answers
            normalized_gen = normalize_answer(generated_response)
            normalized_gold_aliases = normalize_gold_answers(true_answer)
            normalized_gold = normalized_gold_aliases[0]

            # Remove stopwords
            normalized_gen = remove_stop_words(normalized_gen, stop_word_list)

            # Exact Match
            exact_match, partial_match, normalized_gold_aliases = exact_or_contained_match(
                normalized_gen,
                normalized_gold_aliases,
            )
            if exact_match:
                em_score += 1
            elif partial_match:
                partial_match_score += 1

            # NEW CODE by Rishikesh: collect per-example data for the
            # task-specific quality measures (scored/aggregated after the loop).
            if use_rauq:
                try:
                    uncertainty = compute_sequence_uncertainty(
                        model=llm.model,
                        tokenizer=llm.tokenizer,
                        prompt_text=prompt_w_ctx,
                        generated_text=raw_generated_response,
                        device=llm.device,
                    )
                except Exception:
                    uncertainty = float("nan")
                per_example_uncertainty.append(uncertainty)
                per_example_correctness.append(int(exact_match))

            if use_comet:
                comet_sources.append(prompt_w_ctx)
                comet_hyps.append(generated_response)
                comet_refs.append(normalized_gold)

            if use_alignscore:
                alignscore_contexts.append(prompt_w_ctx)
                alignscore_claims.append(generated_response)

            # Keep record (both correct and incorrect)
            incorrect_details.append({
                "Index": index,
                "True Answer": normalized_gold,
                "Gold Aliases": " | ".join(normalized_gold_aliases),
                "Raw Generated Text": raw_generated_response,
                "Cleaned Generated Text": generated_response,
                "Predicted Answer": normalized_gen,
                "EM": int(exact_match),
                "Partial Match": int(partial_match),
            })

            # Compute F1, Precision, Recall
            f1_values = [compute_f1(gold, normalized_gen) for gold in normalized_gold_aliases]
            f1, precision, recall = max(f1_values, key=lambda item: item[0])
            f1_score_total += f1
            precision_total += precision
            recall_total += recall

    # Compute average metrics
    em_score /= total_count
    partial_match_score /= total_count
    avg_f1_score = f1_score_total / total_count
    avg_precision = precision_total / total_count
    avg_recall = recall_total / total_count

    # NEW CODE by Rishikesh: assemble task-specific quality measures.
    quality_metrics = {
        "accuracy": compute_accuracy_qa(em_score),
        "comet": "n/a",
        "alignscore": "n/a",
        "rauq_prr": "n/a",
    }

    if use_comet and comet_hyps:
        comet_mean, _ = compute_comet_scores(comet_sources, comet_hyps, comet_refs)
        quality_metrics["comet"] = comet_mean if comet_mean is not None else "n/a"

    if use_alignscore and alignscore_claims:
        alignscore_mean, _ = compute_alignscore_scores(alignscore_contexts, alignscore_claims)
        quality_metrics["alignscore"] = alignscore_mean if alignscore_mean is not None else "n/a"

    if use_rauq and per_example_uncertainty:
        valid_unc = [u for u in per_example_uncertainty if u == u]  # filter NaN
        mean_uncertainty = float(sum(valid_unc) / len(valid_unc)) if valid_unc else float("nan")
        prr = compute_prr(per_example_uncertainty, per_example_correctness)
        quality_metrics["rauq_prr"] = {
            "mean_uncertainty_nats": mean_uncertainty,
            "prr": prr,
            "n_scored": len(valid_unc),
        }

    return em_score, partial_match_score, avg_f1_score, avg_precision, avg_recall, incorrect_details, quality_metrics


def save_incorrect_details(incorrect_details, name):
    """
    Saves the details of predictions to a CSV file (could include correct or incorrect).
    """
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, f"{name}.csv")
    pd.DataFrame(incorrect_details).to_csv(file_path, index=False)
    print(f"Incorrect predictions saved to {file_path}")
