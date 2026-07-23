# main.py

import os
import argparse

# Set PyTorch CUDA memory allocation configuration to avoid fragmentation
# This should be set before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from utils import (
    set_seed,
    initialize_wandb,
    log_metrics_to_wandb,
    log_metrics_to_wandb_summarization,
    finish_wandb,
    clear_cuda_cache
)
from model import Base_Model
from data_utils import load_data_and_create_prompts
from evaluation import evaluate_model, save_incorrect_details
from evaluation_summarization import evaluate_model_summarization, save_prediction_details


def main(args):
    # 0. Clear CUDA cache at start
    clear_cuda_cache()
    
    # Problem 9 fix - modified by rishikesh.
    # OLD CODE:
    # set_seed(42)
    #
    # NEW CODE:
    # Expose the random seed as an argument so experiments can be reproduced
    # from the command line and reported in the paper table.
    set_seed(args.seed)
    
    # Set default max_new_tokens based on task type
    if args.max_new_tokens is None:
        args.max_new_tokens = 150 if args.task_type == 'summarization' else 25
    
    print(f"Task Type: {args.task_type}")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Precision: {args.precision} (no 4-bit/8-bit quantization)")
    print(f"Seed: {args.seed}")
    print(f"Max New Tokens: {args.max_new_tokens}") 

    # 2. Initialize model
    llm = Base_Model(
        model_name=args.model_name,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        precision=args.precision,
    )
    
    # Clear cache after model loading
    clear_cuda_cache()

    # 3. Define stop words based on task type
    if args.task_type == 'summarization':
        stop_word_list = ["Article:", "Document:", "Dialogue:", "\n\n\n", "\n\n##"]
    else:
        stop_word_list = ["Q:", "Supporting information:", "\n", "\n\n##"]
    llm.set_stop_words(stop_word_list)

    # 4. Load dataset & create prompts
    prompts_with_context, prompts_without_context, references = load_data_and_create_prompts(args)

    # Optionally subsample for memory/speed
    if getattr(args, "max_examples", None) is not None:
        max_n = min(args.max_examples, len(prompts_with_context))
        prompts_with_context = prompts_with_context[:max_n]
        prompts_without_context = prompts_without_context[:max_n]
        references = references[:max_n]

    # 5. Evaluate the model based on task type
    try:
        if args.task_type == 'summarization':
            # Summarization evaluation
            # NEW CODE by Rishikesh: evaluate_model_summarization now also
            # returns a quality_metrics dict (Accuracy/COMET/AlignScore/RAUQ-PRR).
            (rouge1_f, rouge1_p, rouge1_r,
             rouge2_f, rouge2_p, rouge2_r,
             rougel_f, rougel_p, rougel_r,
             prediction_details,
             quality_metrics) = evaluate_model_summarization(
                llm,
                prompts_with_context,
                prompts_without_context,
                references,
                stop_word_list,
                args
            )

            # 6. Save prediction details
            save_prediction_details(prediction_details, args.name)

            # 7. Initialize WandB
            initialize_wandb(args)

            # 8. Log metrics to WandB
            log_metrics_to_wandb_summarization(
                rouge1_f, rouge1_p, rouge1_r,
                rouge2_f, rouge2_p, rouge2_r,
                rougel_f, rougel_p, rougel_r
            )

            # 9. Print results
            print("\n" + "="*50)
            print("SUMMARIZATION EVALUATION RESULTS")
            print("="*50)
            print(f"ROUGE-1 F1: {rouge1_f:.4f} | Precision: {rouge1_p:.4f} | Recall: {rouge1_r:.4f}")
            print(f"ROUGE-2 F1: {rouge2_f:.4f} | Precision: {rouge2_p:.4f} | Recall: {rouge2_r:.4f}")
            print(f"ROUGE-L F1: {rougel_f:.4f} | Precision: {rougel_p:.4f} | Recall: {rougel_r:.4f}")
            # NEW CODE by Rishikesh: task-specific quality measures
            print(f"Accuracy (ROUGE-L>={0.30:.2f} proxy): {quality_metrics.get('accuracy', 'n/a')}")
            print(f"COMET: {quality_metrics.get('comet', 'n/a')}")
            print(f"AlignScore: {quality_metrics.get('alignscore', 'n/a')}")
            rauq_result = quality_metrics.get('rauq_prr', 'n/a')
            if isinstance(rauq_result, dict):
                print(f"RAUQ mean uncertainty (nats): {rauq_result.get('mean_uncertainty_nats')}")
                print(f"RAUQ PRR: {rauq_result.get('prr')}")
                print(f"RAUQ n_scored: {rauq_result.get('n_scored')}")
            else:
                print(f"RAUQ/PRR: {rauq_result}")
            print("="*50 + "\n")

        else:
            # QA evaluation.
            # NEW CODE by Rishikesh: evaluate_model now also returns a
            # quality_metrics dict (Accuracy/COMET/AlignScore/RAUQ-PRR).
            em_score, partial_match, avg_f1, avg_precision, avg_recall, incorrect_details, quality_metrics = evaluate_model(
                llm,
                prompts_with_context,
                prompts_without_context,
                references,
                stop_word_list,
                args
            )

            # 6. Save incorrect predictions
            save_incorrect_details(incorrect_details, args.name)

            # 7. Initialize WandB
            initialize_wandb(args)

            # 8. Log metrics to WandB
            log_metrics_to_wandb(em_score, partial_match, avg_f1, avg_precision, avg_recall)

            # 8b. Print results to console (mirrors the summarization report block below,
            # and matches the "wandb: <metric>" report style used across the project)
            print("\n" + "="*50)
            print("QA EVALUATION RESULTS")
            print("="*50)
            print(f"Run name: {args.name}")
            print(f"Model: {args.model_name} | Dataset: {args.dataset} | Mode: {args.mode}")
            print(f"alpha={args.alpha}, layer_alpha={args.layer_alpha}, start_layer={args.start_layer}")
            print(f"wandb:   Exact Match Score {em_score:.5g}")
            print(f"wandb:  Partial Match Score {partial_match:.5g}")
            print(f"wandb:            F1 Score {avg_f1:.5g}")
            print(f"wandb:     Precision Score {avg_precision:.5g}")
            print(f"wandb:        Recall Score {avg_recall:.5g}")
            # NEW CODE by Rishikesh: task-specific quality measures
            print(f"wandb:   Accuracy Score {quality_metrics.get('accuracy', em_score):.5g}")
            print(f"wandb:   COMET Score {quality_metrics.get('comet', 'n/a')}")
            print(f"wandb: AlignScore Score {quality_metrics.get('alignscore', 'n/a')}")
            rauq_result = quality_metrics.get('rauq_prr', 'n/a')
            if isinstance(rauq_result, dict):
                print(f"wandb: RAUQ mean uncertainty (nats) {rauq_result.get('mean_uncertainty_nats'):.5g}")
                print(f"wandb: RAUQ PRR {rauq_result.get('prr'):.5g}")
                print(f"wandb: RAUQ n_scored {rauq_result.get('n_scored')}")
            else:
                print(f"wandb: RAUQ/PRR {rauq_result}")
            print("="*50 + "\n")

            # 8c. Append a machine-readable summary row so notebook cells can be
            # aggregated later into one results table (results/qa_results_summary.jsonl)
            import json as _json
            _summary_path = os.path.join("results", "qa_results_summary.jsonl")
            os.makedirs("results", exist_ok=True)
            with open(_summary_path, "a") as _f:
                _f.write(_json.dumps({
                    "name": args.name,
                    "model": args.model_name,
                    "dataset": args.dataset,
                    "mode": args.mode,
                    "alpha": args.alpha,
                    "layer_alpha": args.layer_alpha,
                    "start_layer": args.start_layer,
                    "max_examples": getattr(args, "max_examples", None),
                    "em": em_score,
                    "partial_match": partial_match,
                    "f1": avg_f1,
                    "precision": avg_precision,
                    "recall": avg_recall,
                    "accuracy": quality_metrics.get("accuracy", em_score),
                    "comet": quality_metrics.get("comet"),
                    "alignscore": quality_metrics.get("alignscore"),
                    "rauq_prr": quality_metrics.get("rauq_prr"),
                }) + "\n")

        # 9. Finish WandB run (no-op if wandb not installed)
        finish_wandb()

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)

    finally:
        # Explicitly free large in-memory objects and clear CUDA cache at the end of a run
        try:
            del prompts_with_context, prompts_without_context, references
        except Exception:
            pass
        clear_cuda_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ------------------------ Model-related arguments
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b",
                        help="Name of the model to use.")
    parser.add_argument("--num-gpus", type=str, default="1",
                        help="Number of GPUs to use.")
    parser.add_argument("--max_gpu_memory", type=int, default=24,
                        help="Maximum GPU memory to allocate.")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda",
                        help="Device to run the model on.")
    parser.add_argument("--precision", type=str, choices=["bf16", "fp16", "fp32"], default="bf16",
                        help="Non-quantized model precision. Use bf16 on A100/H100/L4, fp16 on older CUDA GPUs, fp32 on CPU/debug.")

    # ------------------------ Task-related arguments
    parser.add_argument("--task-type", type=str, choices=['qa', 'summarization'], default='qa',
                        help="Type of task: 'qa' for question answering or 'summarization' for summarization.")
    
    # ------------------------ Data-related arguments
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset to use. QA: hotpot_qa, squad, squad_v2, strategyqa, boolq, triviaqa, natural_questions, openbookqa, arc_challenge, commonsense_qa, truthfulqa, fever, halueval. Summarization: cnn_dailymail, xsum, samsum.")
    parser.add_argument("--strategyqa-path", type=str, default=None,
                        help="Optional local JSON/JSONL file for StrategyQA (used when dataset='strategyqa').")

    # ------------------------ Decoding/Inference-related arguments
    parser.add_argument("--mode", type=str, default='final_layer_context',
                        help="Generation mode (e.g., CAD, DOLA, LACD, ALACD, etc.).")
    parser.add_argument("--alpha", type=float, default=0.30,
                        help="Alpha value for logits adjustment.")
    parser.add_argument("--layer_alpha", type=float, default=0.5,
                        help="Layer alpha for additional adjustments.")
    parser.add_argument("--start_layer", type=int, default=16,
                        help="Starting layer for comparisons.")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Maximum number of new tokens to generate. Default: 25 for QA, 150 for summarization.")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Optional limit on number of evaluation examples (useful to reduce memory/compute).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible runs.")

    # NEW CODE by Rishikesh: task-specific quality measure flags. Each is
    # opt-in since COMET/AlignScore load separate large models, and RAUQ adds
    # one extra forward pass per example. See quality_metrics.py for details
    # and honesty notes on what each metric actually measures here.
    parser.add_argument("--use-rauq", dest="use_rauq", action="store_true", default=False,
                        help="Compute RAUQ-proxy (mean predictive entropy) and Prediction Rejection "
                             "Ratio (PRR) over the eval set.")
    parser.add_argument("--use-comet", dest="use_comet", action="store_true", default=False,
                        help="Compute COMET scores (requires: pip install unbabel-comet; "
                             "downloads Unbabel/wmt22-comet-da checkpoint on first use).")
    parser.add_argument("--use-alignscore", dest="use_alignscore", action="store_true", default=False,
                        help="Compute AlignScore faithfulness scores (requires: pip install alignscore; "
                             "set env var ALIGNSCORE_CKPT_PATH to a downloaded AlignScore checkpoint).")

    # Example of a subset_layers argument if needed
    # parser.add_argument("--subset-layers", type=int, nargs='+', default=None,
    #                     help="Optional list of layers for contrast_layer_context_nocontext_jsd_subset")

    # ------------------------ Logging-related arguments
    parser.add_argument("--name", type=str, default='default',
                        help="Name of the experiment.")
    parser.add_argument("--project", type=str, default='Decoding_Exp_val2',
                        help="WandB project name.")
    parser.add_argument("--group", type=str, help="Group name for the experiment")

    args = parser.parse_args()
    
    main(args)



