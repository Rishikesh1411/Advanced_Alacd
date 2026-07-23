# quality_metrics.py
"""
Task-specific quality measures for ALACD evaluation:
  - Accuracy
  - COMET
  - AlignScore
  - RAUQ / PRR (uncertainty-based hallucination signal)

Design notes / honesty disclosures (read before citing these in the paper):

1. Accuracy
   - QA: identical to Exact Match (already computed in evaluation.py); exposed
     here again so it appears in the same quality_metrics dict as the other three.
   - Summarization: there's no canonical "accuracy" for free-form summarization.
     We use a threshold-based proxy: an example counts as "accurate" if its
     ROUGE-L F1 exceeds `SUMMARIZATION_ACCURACY_ROUGE_THRESHOLD` (default 0.30).
     This is a proxy, not a standard metric — call it out as such if it goes in
     a paper table.

2. COMET
   - Uses the real `unbabel-comet` package (pip install unbabel-comet) with the
     public `Unbabel/wmt22-comet-da` checkpoint. COMET was built for MT quality
     estimation (source, hypothesis, reference triples), and is being repurposed
     here as a semantic-adequacy score between the generated text and the gold
     reference (source = question+context for QA, source = article for
     summarization). It is NOT a hallucination-specific metric on its own —
     treat it as a fluency/adequacy signal, not a faithfulness signal.
   - Requires downloading a ~1.1GB checkpoint from HuggingFace on first use.
   - If the package or checkpoint isn't available, returns None / 'n/a'
     gracefully rather than crashing your run.

3. AlignScore
   - Uses the real `alignscore` package (pip install alignscore), which IS
     designed for factual consistency / faithfulness checking (context vs.
     claim), so it's the more appropriate metric of the two for hallucination
     work. Requires its own checkpoint download (AlignScore-base, ~1.3GB) and
     you must point ALIGNSCORE_CKPT_PATH at the downloaded .ckpt file (the
     package does not auto-download it).
   - If the package or checkpoint path isn't available, returns None / 'n/a'.

4. RAUQ / PRR
   - RAUQ here is a simplified PROXY: mean per-token predictive entropy
     computed via one teacher-forced forward pass over (prompt + generated
     text). It is NOT the exact attention-lookback-head algorithm from the
     "Efficient Hallucination Detection for LLMs Using Uncertainty-Aware
     Attention" paper (that requires hooking a specific attention head during
     generation itself — a bigger change to model.py's decode loops).
   - PRR (Prediction Rejection Ratio, Malinin & Gales style) is computed
     properly given the per-example uncertainty + correctness pairs.

Both COMET and AlignScore are opt-in via CLI flags (--use-comet / --use-alignscore)
because they load large separate models and will slow down / add VRAM pressure
to a run that's already loading a 7B-13B causal LM. RAUQ is opt-in via
--use-rauq (cheap: reuses the already-loaded model, one extra forward pass
per example).
"""

import os
import torch
import torch.nn.functional as F
import numpy as np

SUMMARIZATION_ACCURACY_ROUGE_THRESHOLD = 0.30

# Optional env var pointing at a downloaded AlignScore checkpoint (.ckpt file).
# The alignscore package does not auto-download weights.
ALIGNSCORE_CKPT_PATH = os.environ.get("ALIGNSCORE_CKPT_PATH", None)
ALIGNSCORE_MODEL_NAME = os.environ.get("ALIGNSCORE_MODEL_NAME", "roberta-large")

# Lazily-initialized singletons so we don't reload COMET/AlignScore per-example.
_comet_model = None
_alignscore_scorer = None


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def compute_accuracy_qa(em_score):
    """QA accuracy is just Exact Match, already computed by evaluation.py."""
    return em_score


def compute_accuracy_summarization(rougel_f_per_example, threshold=SUMMARIZATION_ACCURACY_ROUGE_THRESHOLD):
    """
    Proxy accuracy for summarization: fraction of examples whose ROUGE-L F1
    exceeds `threshold`. rougel_f_per_example: list[float].
    """
    if not rougel_f_per_example:
        return float("nan")
    hits = sum(1 for score in rougel_f_per_example if score >= threshold)
    return hits / len(rougel_f_per_example)


# ---------------------------------------------------------------------------
# COMET
# ---------------------------------------------------------------------------

def _load_comet_model():
    global _comet_model
    if _comet_model is not None:
        return _comet_model
    try:
        from comet import download_model, load_from_checkpoint
        model_path = download_model("Unbabel/wmt22-comet-da")
        _comet_model = load_from_checkpoint(model_path)
    except Exception as e:
        print(f"[quality_metrics] COMET unavailable ({e}). "
              f"Install with: pip install unbabel-comet. Skipping COMET scoring.")
        _comet_model = False
    return _comet_model


def compute_comet_scores(sources, hypotheses, references, gpus=1 if torch.cuda.is_available() else 0):
    """
    sources, hypotheses, references: parallel lists[str].
    Returns (mean_score: float|None, per_example_scores: list|None).
    """
    model = _load_comet_model()
    if not model:
        return None, None
    try:
        data = [
            {"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)
        ]
        output = model.predict(data, batch_size=8, gpus=gpus, progress_bar=False)
        scores = list(output["scores"]) if "scores" in output else list(output.scores)
        return float(np.mean(scores)), scores
    except Exception as e:
        print(f"[quality_metrics] COMET scoring failed ({e}). Skipping.")
        return None, None


# ---------------------------------------------------------------------------
# AlignScore
# ---------------------------------------------------------------------------

def _load_alignscore_scorer():
    global _alignscore_scorer
    if _alignscore_scorer is not None:
        return _alignscore_scorer
    if not ALIGNSCORE_CKPT_PATH or not os.path.exists(ALIGNSCORE_CKPT_PATH):
        print("[quality_metrics] AlignScore checkpoint not found. Set env var "
              "ALIGNSCORE_CKPT_PATH to a downloaded AlignScore .ckpt file "
              "(see https://github.com/yuh-zha/AlignScore for download links). "
              "Skipping AlignScore.")
        _alignscore_scorer = False
        return _alignscore_scorer
    try:
        from alignscore import AlignScore
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _alignscore_scorer = AlignScore(
            model=ALIGNSCORE_MODEL_NAME,
            batch_size=16,
            device=device,
            ckpt_path=ALIGNSCORE_CKPT_PATH,
            evaluation_mode="nli_sp",
        )
    except Exception as e:
        print(f"[quality_metrics] AlignScore unavailable ({e}). "
              f"Install with: pip install alignscore. Skipping AlignScore scoring.")
        _alignscore_scorer = False
    return _alignscore_scorer


def compute_alignscore_scores(contexts, claims):
    """
    contexts: list[str] — the source text the claim should be faithful to
              (question+context for QA, article for summarization).
    claims:   list[str] — the generated text to check for faithfulness.
    Returns (mean_score: float|None, per_example_scores: list|None).
    """
    scorer = _load_alignscore_scorer()
    if not scorer:
        return None, None
    try:
        scores = scorer.score(contexts=contexts, claims=claims)
        return float(np.mean(scores)), list(scores)
    except Exception as e:
        print(f"[quality_metrics] AlignScore scoring failed ({e}). Skipping.")
        return None, None


# ---------------------------------------------------------------------------
# RAUQ (proxy) + PRR
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_sequence_uncertainty(model, tokenizer, prompt_text, generated_text, device, max_length=2048):
    """
    Teacher-forces (prompt_text + generated_text) through the model once and
    computes mean predictive entropy (nats) over ONLY the generated token
    positions. Higher = more uncertain / more hallucination-prone.

    Simplified RAUQ proxy — see module docstring for the caveat vs. the exact
    published attention-lookback algorithm.
    """
    if not generated_text or len(generated_text.strip()) == 0:
        return float("nan")

    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", truncation=True, max_length=max_length
    ).input_ids.to(device)
    full_ids = tokenizer(
        prompt_text + generated_text, return_tensors="pt", truncation=True, max_length=max_length
    ).input_ids.to(device)

    gen_len = full_ids.shape[1] - prompt_ids.shape[1]
    if gen_len <= 0:
        return float("nan")

    outputs = model(full_ids)
    logits = outputs.logits[0]  # (seq_len, vocab)

    start = prompt_ids.shape[1] - 1
    end = full_ids.shape[1] - 1
    gen_logits = logits[start:end]  # (gen_len, vocab)

    probs = F.softmax(gen_logits.float(), dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-12))
    entropy = -(probs * log_probs).sum(dim=-1)  # (gen_len,)

    return entropy.mean().item()


def compute_prr(uncertainties, correctness_flags, n_random_perms=20, seed=42):
    """
    Prediction Rejection Ratio (Malinin & Gales style).
    PRR = (AUC_model - AUC_random) / (AUC_oracle - AUC_random)
    """
    uncertainties = np.asarray(uncertainties, dtype=float)
    correctness = np.asarray(correctness_flags, dtype=float)

    valid = ~np.isnan(uncertainties)
    uncertainties = uncertainties[valid]
    correctness = correctness[valid]

    n = len(uncertainties)
    if n < 2:
        return float("nan")

    def rejection_curve_auc(order):
        sorted_correct = correctness[order]
        accs = np.empty(n)
        for k in range(n):
            remaining = sorted_correct[k:]
            accs[k] = remaining.mean() if len(remaining) > 0 else 1.0
        return accs.mean()

    model_order = np.argsort(-uncertainties)
    auc_model = rejection_curve_auc(model_order)

    rng = np.random.default_rng(seed)
    random_aucs = [rejection_curve_auc(rng.permutation(n)) for _ in range(n_random_perms)]
    auc_random = float(np.mean(random_aucs))

    oracle_order = np.argsort(correctness)
    auc_oracle = rejection_curve_auc(oracle_order)

    denom = auc_oracle - auc_random
    if abs(denom) < 1e-9:
        return float("nan")

    return float((auc_model - auc_random) / denom)
