# evaluation_summarization.py

import os
import torch
import pandas as pd
from tqdm import tqdm

from utils import (
    clear_cuda_cache,
    compute_rouge_scores
)

# NEW CODE by Rishikesh: task-specific quality measures for summarization.
from quality_metrics import (
    compute_accuracy_summarization,
    compute_comet_scores,
    compute_alignscore_scores,
    compute_sequence_uncertainty,
    compute_prr,
    SUMMARIZATION_ACCURACY_ROUGE_THRESHOLD,
)


def evaluate_model_summarization(llm, prompts_with_context, prompts_without_context, reference_summaries, stop_word_list, args):
    """
    Evaluates the model on summarization tasks using ROUGE metrics.
    Also returns a quality_metrics dict with Accuracy (ROUGE-L threshold
    proxy), COMET, AlignScore, and RAUQ/PRR (each opt-in via
    --use-comet / --use-alignscore / --use-rauq).
    """
    rouge1_f_total = 0
    rouge1_p_total = 0
    rouge1_r_total = 0
    rouge2_f_total = 0
    rouge2_p_total = 0
    rouge2_r_total = 0
    rougel_f_total = 0
    rougel_p_total = 0
    rougel_r_total = 0
    total_count = len(prompts_with_context)

    use_rauq = bool(getattr(args, "use_rauq", False))
    use_comet = bool(getattr(args, "use_comet", False))
    use_alignscore = bool(getattr(args, "use_alignscore", False))

    rougel_f_per_example = []
    per_example_uncertainty = []
    per_example_correctness = []  # proxy correctness = rougeL_f >= threshold, for PRR
    comet_sources, comet_hyps, comet_refs = [], [], []
    alignscore_contexts, alignscore_claims = [], []

    prediction_details = []

    # Use inference_mode for better memory efficiency
    with torch.inference_mode():
        for index, (prompt_w_ctx, prompt_no_ctx, reference_summary) in enumerate(
            tqdm(zip(prompts_with_context, prompts_without_context, reference_summaries), total=total_count)
        ):
            # Clear cache before processing each sample
            if index % 5 == 0 and torch.cuda.is_available():
                clear_cuda_cache()
            
            # Multi-strategy generation: Try multiple methods and pick the best
            max_new_tokens = getattr(args, 'max_new_tokens', 150)  # Longer for summarization
            candidates = []
            
            # Strategy 1: Primary mode
            try:
                primary_response = llm.generate(
                    input_text=prompt_w_ctx,
                    input_text2=prompt_no_ctx,
                    mode=args.mode,
                    alpha=args.alpha,
                    layer_alpha=args.layer_alpha,
                    start_layer=args.start_layer,
                    max_new_tokens=max_new_tokens
                )
                if primary_response and primary_response.strip():
                    candidates.append(primary_response.strip())
            except Exception as e:
                print(f"Error in primary generation: {e}", flush=True)
                pass
            
            # Strategy 2: Baseline (more stable)
            if args.mode != "final_layer_context":
                try:
                    baseline_response = llm.generate(
                        input_text=prompt_w_ctx,
                        input_text2=prompt_no_ctx,
                        mode="final_layer_context",
                        alpha=args.alpha,
                        layer_alpha=args.layer_alpha,
                        start_layer=args.start_layer,
                        max_new_tokens=max_new_tokens
                    )
                    if baseline_response and baseline_response.strip():
                        candidates.append(baseline_response.strip())
                except Exception as e:
                    print(f"Error in baseline generation: {e}", flush=True)
                    pass
            
            # Strategy 3: CAD mode (if not already used)
            if args.mode not in ["CAD", "final_layer_context"]:
                try:
                    cad_response = llm.generate(
                        input_text=prompt_w_ctx,
                        input_text2=prompt_no_ctx,
                        mode="CAD",
                        alpha=args.alpha,
                        layer_alpha=args.layer_alpha,
                        start_layer=args.start_layer,
                        max_new_tokens=max_new_tokens
                    )
                    if cad_response and cad_response.strip():
                        candidates.append(cad_response.strip())
                except Exception as e:
                    print(f"Error in CAD generation: {e}", flush=True)
                    pass
            
            # Select best candidate (for now, use the first non-empty one)
            # In future, could use ROUGE to select best candidate
            generated_summary = ""
            if candidates:
                # Use the primary response if available, otherwise first candidate
                generated_summary = candidates[0] if candidates else ""
            else:
                # Fallback: try one more time with baseline
                try:
                    fallback_response = llm.generate(
                        input_text=prompt_w_ctx,
                        input_text2=prompt_no_ctx,
                        mode="final_layer_context",
                        alpha=args.alpha,
                        layer_alpha=args.layer_alpha,
                        start_layer=args.start_layer,
                        max_new_tokens=max_new_tokens
                    )
                    if fallback_response and fallback_response.strip():
                        generated_summary = fallback_response.strip()
                except:
                    pass
            
            # Clean the generated summary
            generated_summary = generated_summary.strip()
            
            # Compute ROUGE scores
            if generated_summary and reference_summary:
                rouge_scores = compute_rouge_scores(reference_summary, generated_summary)
                
                rouge1_f_total += rouge_scores['rouge1']['f']
                rouge1_p_total += rouge_scores['rouge1']['p']
                rouge1_r_total += rouge_scores['rouge1']['r']
                rouge2_f_total += rouge_scores['rouge2']['f']
                rouge2_p_total += rouge_scores['rouge2']['p']
                rouge2_r_total += rouge_scores['rouge2']['r']
                rougel_f_total += rouge_scores['rougel']['f']
                rougel_p_total += rouge_scores['rougel']['p']
                rougel_r_total += rouge_scores['rougel']['r']

                # NEW CODE by Rishikesh: track for Accuracy proxy + RAUQ/COMET/AlignScore
                this_rougel_f = rouge_scores['rougel']['f']
            else:
                # If generation failed, all scores are 0
                this_rougel_f = 0.0

            rougel_f_per_example.append(this_rougel_f)

            # NEW CODE by Rishikesh: collect per-example data for task-specific
            # quality measures (scored/aggregated after the loop).
            if use_rauq:
                try:
                    uncertainty = compute_sequence_uncertainty(
                        model=llm.model,
                        tokenizer=llm.tokenizer,
                        prompt_text=prompt_w_ctx,
                        generated_text=generated_summary,
                        device=llm.device,
                    )
                except Exception:
                    uncertainty = float("nan")
                per_example_uncertainty.append(uncertainty)
                # Proxy "correctness" for PRR on summarization: did this example
                # clear the same ROUGE-L threshold used for the Accuracy proxy?
                per_example_correctness.append(int(this_rougel_f >= SUMMARIZATION_ACCURACY_ROUGE_THRESHOLD))

            if use_comet:
                comet_sources.append(prompt_w_ctx)
                comet_hyps.append(generated_summary)
                comet_refs.append(reference_summary)

            if use_alignscore:
                # Faithfulness of the summary to the source article
                alignscore_contexts.append(prompt_w_ctx)
                alignscore_claims.append(generated_summary)
            
            # Keep record
            prediction_details.append({
                "Index": index,
                "Reference Summary": reference_summary[:200] + "..." if len(reference_summary) > 200 else reference_summary,
                "Generated Summary": generated_summary[:200] + "..." if len(generated_summary) > 200 else generated_summary
            })
            
            # Clear cache after processing each sample
            if torch.cuda.is_available():
                clear_cuda_cache()

    # Clear cache one final time
    if torch.cuda.is_available():
        clear_cuda_cache()
    
    # Compute average metrics
    avg_rouge1_f = rouge1_f_total / total_count if total_count > 0 else 0
    avg_rouge1_p = rouge1_p_total / total_count if total_count > 0 else 0
    avg_rouge1_r = rouge1_r_total / total_count if total_count > 0 else 0
    avg_rouge2_f = rouge2_f_total / total_count if total_count > 0 else 0
    avg_rouge2_p = rouge2_p_total / total_count if total_count > 0 else 0
    avg_rouge2_r = rouge2_r_total / total_count if total_count > 0 else 0
    avg_rougel_f = rougel_f_total / total_count if total_count > 0 else 0
    avg_rougel_p = rougel_p_total / total_count if total_count > 0 else 0
    avg_rougel_r = rougel_r_total / total_count if total_count > 0 else 0

    # NEW CODE by Rishikesh: assemble task-specific quality measures.
    quality_metrics = {
        "accuracy": compute_accuracy_summarization(rougel_f_per_example),
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
        valid_unc = [u for u in per_example_uncertainty if u == u]
        mean_uncertainty = float(sum(valid_unc) / len(valid_unc)) if valid_unc else float("nan")
        prr = compute_prr(per_example_uncertainty, per_example_correctness)
        quality_metrics["rauq_prr"] = {
            "mean_uncertainty_nats": mean_uncertainty,
            "prr": prr,
            "n_scored": len(valid_unc),
        }

    return (
        avg_rouge1_f, avg_rouge1_p, avg_rouge1_r,
        avg_rouge2_f, avg_rouge2_p, avg_rouge2_r,
        avg_rougel_f, avg_rougel_p, avg_rougel_r,
        prediction_details,
        quality_metrics
    )


def save_prediction_details(prediction_details, name):
    """
    Saves the details of predictions to a CSV file.
    """
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, f"{name}.csv")
    pd.DataFrame(prediction_details).to_csv(file_path, index=False)
    print(f"Prediction details saved to {file_path}")

