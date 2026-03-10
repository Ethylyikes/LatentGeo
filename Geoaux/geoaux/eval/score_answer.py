"""Answer scoring: exact match + optional LLM judge."""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from geoaux.utils import (
    get_api_key,
    get_base_url,
    get_default_model,
    load_config,
    read_json,
    save_json,
)

# ---------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------

JUDGE_PROMPT = """
You are a math teacher grading a student's answer.
Question: {question}
Correct Answer: {ground_truth}
Student Answer: {prediction}

Does the student answer match the correct answer?
For numerical answers, they should be equivalent (e.g., 1.5 equals 3/2).
For multiple choice, the option letter should match.
For expressions, they should be algebraically equivalent.

Reply only with "True" or "False".
"""


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def safe_equal(prediction, answer):
    if prediction is None:
        prediction = ""
    if answer is None:
        answer = ""

    def normalize(s):
        s = str(s).strip().lower()
        s = "".join(s.split())
        if s.endswith("."):
            s = s[:-1]
        s = s.replace("dfrac", "frac")
        return s

    pred, ans = normalize(prediction), normalize(answer)
    if pred == ans:
        return True

    try:
        def to_float(val):
            if "/" in val:
                n, d = val.split("/")
                return float(n) / float(d)
            return float(val)
        if abs(to_float(pred) - to_float(ans)) < 1e-6:
            return True
    except Exception:
        pass

    return False


class LLMJudge:
    def __init__(self, api_key, base_url, model_name):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def judge(self, question, ground_truth, prediction):
        prompt = JUDGE_PROMPT.format(
            question=question, ground_truth=ground_truth, prediction=prediction
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=10,
            )
            return "true" in resp.choices[0].message.content.strip().lower()
        except Exception as e:
            print(f"Error calling LLM Judge: {e}")
            return False


def score_single_problem(problem, meta_map, judge, use_judge):
    prediction = problem.get("normalized_extraction", "")
    ground_truth = problem.get("correct_answer", "")
    text_input = problem.get("text_input", "")

    pid = problem.get("id")
    if pid and pid in meta_map:
        problem.update(meta_map[pid])

    is_correct = safe_equal(prediction, ground_truth)

    if not is_correct and use_judge and prediction:
        is_correct = judge.judge(text_input, ground_truth, prediction)

    res = problem.copy()
    res["is_correct"] = is_correct
    return res


def _build_meta_map(benchmark_file):
    meta_map = {}
    if not benchmark_file or not os.path.exists(benchmark_file):
        return meta_map
    try:
        bench_data = read_json(benchmark_file)
        for item in bench_data:
            if "id" not in item:
                continue
            visual_ops = item.get("Visual Operation", [])
            if isinstance(visual_ops, list):
                visual_ops = [re.sub(r"（\d+）", "", op).strip() for op in visual_ops]
            else:
                visual_ops = []
            meta_map[item["id"]] = {
                "knowledge": item.get("knowledge", "Unknown"),
                "subknowledge": item.get("subknowledge", "Unknown"),
                "Visual Operation": visual_ops,
            }
    except Exception as e:
        print(f"Warning: Failed to load benchmark metadata: {e}")
    return meta_map


def _print_report(results, output_dir):
    df = pd.DataFrame(results)
    total = len(df)
    correct = df["is_correct"].sum()
    accuracy = correct / total if total > 0 else 0

    lines = [
        "\nBenchmark Evaluation Results",
        "=============================================",
        f"ALL: Accuracy: {accuracy:.4f} (count: {total})",
        "",
    ]

    if "knowledge" in df.columns:
        lines.append("--- Accuracy by Knowledge ---")
        for k_type, group in df.groupby("knowledge"):
            acc = group["is_correct"].mean()
            lines.append(f"{k_type:<35}: {acc:.4f} (count: {len(group)})")
        lines.append("")

    if "Visual Operation" in df.columns:
        lines.append("--- Accuracy by Visual Operation ---")
        df_exp = df.explode("Visual Operation")
        df_exp = df_exp[df_exp["Visual Operation"].notna() & (df_exp["Visual Operation"] != "")]
        for op, group in df_exp.groupby("Visual Operation"):
            acc = group["is_correct"].mean()
            lines.append(f"{op:<45}: {acc:.4f} (count: {len(group)})")
        lines.append("")

    report = "\n".join(lines)
    print(report)

    report_file = os.path.join(output_dir, "score_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved report to {report_file}")


def run_scoring(
    input_file,
    output_dir=None,
    api_key=None,
    base_url=None,
    model=None,
    benchmark_file=None,
    use_judge=False,
    limit=None,
    num_workers=16,
):
    cfg = load_config()
    output_dir = output_dir or cfg.get("eval_output_dir", "./eval_results")
    api_key = api_key or cfg["api"].get("api_key") or get_api_key()
    base_url = base_url or cfg["api"].get("base_url", "") or get_base_url("")
    model = model or cfg["api"].get("default_model", get_default_model("qwen-plus"))
    benchmark_file = benchmark_file or cfg.get("data_path", "")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {input_file}...")
    data = read_json(input_file)
    if limit and limit > 0:
        data = data[:limit]

    meta_map = _build_meta_map(benchmark_file)

    judge = None
    if use_judge:
        if not api_key:
            raise ValueError(
                "Missing API key for judge scoring. Set DASHSCOPE_API_KEY "
                "(or GEOAUX_API_KEY), pass --api_key, or disable --use_judge."
            )
        print(f"Initializing Judge: {model} at {base_url}")
        judge = LLMJudge(api_key, base_url, model)

    results = []
    print(f"Scoring answers with {num_workers} workers...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(score_single_problem, p, meta_map, judge, use_judge): p
            for p in data
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring"):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error scoring item: {e}")

    try:
        results.sort(key=lambda x: int(x["id"]) if str(x["id"]).isdigit() else str(x["id"]))
    except Exception:
        pass

    output_file = os.path.join(output_dir, "scored_results.json")
    save_json(results, output_file)
    print(f"Saved scored results to {output_file}")

    _print_report(results, output_dir)
    return output_file
