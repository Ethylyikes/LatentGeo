
from __future__ import annotations

import os
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch
from mathruler.grader import extract_boxed_content, grade_answer
from transformers import PreTrainedTokenizer

from verl.workers.rollout.utils.util import extract_no_boxed_answer  # reuse helper
from tools.api_judge import api_batch_judge

BOXED_STRICT_RE = re.compile(r"\\boxed\{.+?\}", re.DOTALL | re.IGNORECASE)


def _latent_token_pair() -> tuple[str, str]:
    start_tok = os.environ.get("LATENT_START_TOKEN") or os.environ.get("ABS_VIS_START_TOKEN") or "<|latent_start|>"
    end_tok = os.environ.get("LATENT_END_TOKEN") or os.environ.get("ABS_VIS_END_TOKEN") or "<|latent_end|>"
    return start_tok, end_tok


def _clean_text(s: str) -> str:
    s = re.sub(r"<\|.*?\|>", " ", s)  # drop special tokens
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _extract_answer(resp: str) -> str:
    ans = extract_boxed_content(resp)
    if ans == "None" or ans is None:
        ans = extract_no_boxed_answer(resp) or ""
    return _clean_text(ans)


def _eval_numeric(expr: str):
    try:
        import sympy as sp
        return float(sp.N(sp.sympify(expr)))
    except Exception:
        try:
            return float(expr)
        except Exception:
            return None


def _mcq_match(pred_ans: str, gt_ans: str) -> Optional[float]:
    letters = {"A", "B", "C", "D"}
    if gt_ans.upper() in letters:
        return 1.0 if pred_ans.upper().strip()[:1] == gt_ans.upper() else 0.0
    return None


def _strict_equiv(pred_ans: str, gt_ans: str) -> bool:
    try:
        return bool(grade_answer(pred_ans, gt_ans))
    except Exception:
        pass
    return pred_ans.strip().lower() == gt_ans.strip().lower()


def _accuracy_against_target(pred_ans: str, gt_ans: str) -> float:
    gt_clean = _clean_text(gt_ans or "")

    mcq = _mcq_match(pred_ans, gt_clean)
    if mcq is not None:
        return mcq

    if _strict_equiv(pred_ans, gt_clean):
        return 1.0

    pred_num = _eval_numeric(pred_ans)
    gt_num = _eval_numeric(gt_clean)
    if pred_num is not None and gt_num is not None:
        tol = 0.02 * abs(gt_num)
        return 1.0 if abs(pred_num - gt_num) <= tol else 0.0

    return 0.0


def accuracy_component(pred: str, gt: str, answer_label: Optional[str] = None) -> float:
    """
    Compute accuracy against the main ground-truth and, if provided, an optional choice label.
    Succeeds when either comparison matches under the existing logic.
    """
    pred_ans = _extract_answer(pred or "")
    acc = _accuracy_against_target(pred_ans, gt)

    if answer_label is not None:
        acc = max(acc, _accuracy_against_target(pred_ans, str(answer_label)))

    return acc


def format_component(pred: str) -> float:
    # Lenient: reward if the text contains a boxed answer marker (with or without escaped backslash).
    if not pred:
        return 0.0
    p = pred.lower()
    if "boxed{" in p or "\boxed" in p:
        return 0.2
    return 0.0


def _count_latent_segments(pred: str) -> int:
    start_tok, end_tok = _latent_token_pair()
    p = pred or ""
    start_count = p.count(start_tok)
    end_count = p.count(end_tok)

    # Only a single start and single end can ever yield a "1" (reward). Anything else is penalized.
    if start_count == 1 and end_count == 1:
        try:
            pattern = re.compile(rf"{re.escape(start_tok)}(.*?){re.escape(end_tok)}", re.DOTALL)
        except re.error:
            return 0
        return 1 if pattern.search(p) else 2  # mis-ordered tokens count as invalid

    if start_count == 0 and end_count == 0:
        return 0

    # Imbalanced or extra markers are treated as multiple segments to trigger penalty.
    return max(start_count, end_count, 2)


def latent_component(pred: str, label: Optional[str]) -> float:
    """
    Reward exactly one latent segment with +0.5; penalize missing or multiple segments by -0.2.
    Ignores label; always expect latent markers.
    """
    _ = label  # kept for signature compatibility; label is ignored.
    latent_segments = _count_latent_segments(pred)
    if latent_segments == 1:
        return 0.4
    if latent_segments >= 2:
        return -0.2
    return -0.2


def length_component(
    pred: str,
    resp_len: Optional[int],
    tokenizer: Optional[PreTrainedTokenizer],
    target_length: int = 900,
    max_reward: float = 0.2,
) -> float:
    """
    Penalize responses that exceed 90% of the target length; no reward below that point.
    Penalty scales linearly from 0 at 0.9 * target_length down to -max_reward (default -0.2)
    at and beyond target_length.
    """
    if resp_len is None:
        if tokenizer is not None:
            try:
                resp_len = len(tokenizer.encode(pred or "", add_special_tokens=False))
            except Exception:
                resp_len = 0
        else:
            resp_len = 0

    token_len = max(int(resp_len), 0)
    target_length = max(int(target_length), 1)
    max_reward = max(float(max_reward), 0.0)

    penalty_start = 0.9 * target_length
    if token_len <= penalty_start:
        return 0.0

    over = token_len - penalty_start
    denom = max(target_length - penalty_start, 1e-6)
    severity = min(1.0, over / denom)
    return float(-max_reward * severity)


def _tokenize_for_repetition(pred: str, tokenizer: Optional[PreTrainedTokenizer]) -> List[int]:
    if tokenizer is not None:
        try:
            return tokenizer.encode(pred or "", add_special_tokens=False)
        except Exception:
            pass
    # Fallback: whitespace tokenization.
    return (pred or "").split()


def _ngram_repetition_ratio(tokens: List, n: int) -> float:
    if n <= 0 or len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    dup = sum(c - 1 for c in counts.values() if c > 1)
    return float(max(0.0, dup) / len(ngrams))


def _max_token_run(tokens: List) -> int:
    if not tokens:
        return 0
    longest = 1
    cur = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 1
    return longest


def repetition_penalty_component(
    pred: str,
    tokenizer: Optional[PreTrainedTokenizer],
    *,
    ngram_thresholds: Optional[Dict[int, float]] = None,
    run_length_threshold: int = 8,
    penalty_weight: float = 0.3,
    max_penalty: float = 1.5,
) -> tuple[float, Dict[str, float]]:
    """
    Compute a negative repetition penalty based on n-gram duplication and long runs.
    Returns (penalty, stats).
    """
    tokens = _tokenize_for_repetition(pred or "", tokenizer)
    thresholds = {3: 0.2, 4: 0.15}
    if ngram_thresholds:
        thresholds.update({int(k): float(v) for k, v in ngram_thresholds.items()})

    penalty = 0.0
    stats: Dict[str, float] = {}

    max_ratio = 0.0
    for n, thresh in thresholds.items():
        ratio = _ngram_repetition_ratio(tokens, n)
        max_ratio = max(max_ratio, ratio)
        stats[f"{n}gram_ratio"] = float(ratio)
        if ratio > thresh:
            over = ratio - thresh
            severity = min(1.0, over / max(1e-6, 1 - thresh))
            penalty -= penalty_weight * severity

    max_run = _max_token_run(tokens)
    stats["max_run"] = float(max_run)
    if run_length_threshold > 0 and max_run >= run_length_threshold:
        over_run = max_run - run_length_threshold + 1
        run_severity = min(1.0, over_run / run_length_threshold)
        penalty -= penalty_weight * run_severity

    if max_penalty is not None:
        penalty = max(penalty, -abs(max_penalty))

    stats["penalty"] = float(penalty)
    stats["max_ratio"] = float(max_ratio)
    return float(penalty), stats


def compute_score_wzh(
    predicts: List[str],
    targets: List,
    *,
    labels: Optional[List[str]] = None,
    answer_labels: Optional[List[str]] = None,
    resp_lengths: Optional[torch.Tensor] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    length_reward_target: Optional[int] = None,
    length_reward_max: float = 0.2,
    ref_resp_lengths=None,
    repetition_penalty_weight: float = 0.4,
    repetition_ngram_thresholds: Optional[Dict[int, float]] = None,
    repetition_run_threshold: int = 8,
    repetition_max_penalty: float = 1.5,
    length_penalty_weight: float = 0.0,
    max_response_length: Optional[int] = None,
    max_length: Optional[int] = None,
    **kwargs,
) -> List[Dict[str, float]]:
    labels = labels or [None] * len(predicts)
    if answer_labels is None:
        answer_labels = [None] * len(predicts)
    use_correctness = False
    if len(targets) > 0 and isinstance(targets[0], (int, float, np.floating)):
        use_correctness = True
    if resp_lengths is None:
        resp_lengths_list = [None] * len(predicts)
    else:
        resp_lengths_list = [int(x) for x in resp_lengths]

    # Resolve target length: explicit > dynamic (0.9 * max_response_length or max_length) > fallback.
    resolved_target_length = length_reward_target if length_reward_target and length_reward_target > 0 else None
    if resolved_target_length is None:
        base_length = None
        if max_response_length and max_response_length > 0:
            base_length = max_response_length
        elif max_length and max_length > 0:
            base_length = max_length
        if base_length:
            resolved_target_length = int(max(1, round(0.9 * base_length)))
        else:
            resolved_target_length = 900

    scores = []
    for i, (pred, tgt, label, resp_len) in enumerate(zip(predicts, targets, labels, resp_lengths_list)):
        ans_label = answer_labels[i] if i < len(answer_labels) else None
        if use_correctness:
            acc = float(tgt)
        else:
            acc = accuracy_component(pred, tgt, ans_label)

        fmt = format_component(pred)
        lat = latent_component(pred, label)
        length_reward = length_component(
            pred,
            resp_len,
            tokenizer,
            target_length=resolved_target_length,
            max_reward=length_reward_max,
        )

        base_reward = float(acc + fmt + lat + length_reward)

        repetition_penalty, repetition_stats = repetition_penalty_component(
            pred,
            tokenizer,
            ngram_thresholds=repetition_ngram_thresholds,
            run_length_threshold=repetition_run_threshold,
            penalty_weight=repetition_penalty_weight,
            max_penalty=repetition_max_penalty,
        )

        total = float(base_reward + repetition_penalty)
        scores.append(
            {
                "score": total,
                "accuracy": float(acc),
                "format": float(fmt),
                "latent": float(lat),
                "length": float(length_reward),
                "repetition": float(repetition_penalty),
                "repetition_ratio": float(repetition_stats.get("max_ratio", 0.0)),
                "repetition_max_run": float(repetition_stats.get("max_run", 0.0)),
            }
        )
    return scores


def rule_then_api_batch_judge_wzh(
    questions: List[Optional[str]],
    preds: List[Optional[str]],
    gts: List[Optional[str]],
    *,
    api_name: Optional[str] = "gemini-2.5-pro",
    api_max_workers: int = 32,
    api_kwargs: Optional[Dict] = None,
    client=None,
    repetition_penalty: bool = False,
    labels: Optional[List[str]] = None,
    answer_labels: Optional[List[str]] = None,
):
    correctness = []
    for idx, (pred, gt) in enumerate(zip(preds, gts)):
        ans_label = answer_labels[idx] if answer_labels is not None and idx < len(answer_labels) else None
        correctness.append(float(accuracy_component(pred, gt, ans_label)))

    need_api_idx = [i for i, c in enumerate(correctness) if c == 0.0]
    if not need_api_idx:
        return correctness

    questions_api = [questions[i] for i in need_api_idx]
    preds_api = [preds[i] for i in need_api_idx]
    gts_api = [gts[i] for i in need_api_idx]

    api_correctness_list = api_batch_judge(
        questions_api,
        preds_api,
        gts_api,
        api_name=api_name,
        api_max_workers=api_max_workers,
        api_kwargs=api_kwargs,
        client=client,
        repetition_penalty=repetition_penalty,
    )

    for local_idx, global_idx in enumerate(need_api_idx):
        if api_correctness_list[local_idx] is not None:
            correctness[global_idx] = float(api_correctness_list[local_idx])

    return correctness


def rule_only_batch_judge_wzh(
    questions: List[Optional[str]],
    preds: List[Optional[str]],
    gts: List[Optional[str]],
    *,
    labels: Optional[List[str]] = None,
    answer_labels: Optional[List[str]] = None,
    **kwargs,
):
    """Pure rule-based judge: just accuracy_component per sample, no API fallback."""
    outputs = []
    for idx, (p, g) in enumerate(zip(preds, gts)):
        ans_label = answer_labels[idx] if answer_labels is not None and idx < len(answer_labels) else None
        outputs.append(float(accuracy_component(p, g, ans_label)))
    return outputs
