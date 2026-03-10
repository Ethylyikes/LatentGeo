"""Answer extraction from model responses (regex + optional LLM)."""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

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
# Demo prompt for LLM-based extraction
# ---------------------------------------------------------------------

DEMO_PROMPT = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Guidelines:
1. Extract ONLY the final result. Do not include introductory text like "The answer is" or "Option".
2. For Multiple Choice Questions (if options A, B, C, D are present), extract ONLY the uppercase letter (e.g., "A", "B").
   - If the model outputs the content of the option (e.g., "30 degrees"), try to map it back to the corresponding letter if obvious, otherwise extract the content.
   - Ideally, if the model says "The answer is Option A", just output "A".
3. For Numerical/Expression questions, extract the mathematical value or expression.
   - Remove LaTeX boxing like \\boxed{...}.
   - Keep LaTeX formatting for math expressions (e.g., \\frac{1}{2}, \\sqrt{3}).

Examples:

Hint: Multiple Choice Question
Question: What is 2+2? Options: A. 3, B. 4, C. 5
Model response: The calculation gives 4. Thus the correct option is B.
Extracted answer: B

Hint: Numerical Question
Question: Solve x + 5 = 10.
Model response: Subtracting 5 from both sides, x = 5. \\boxed{5}
Extracted answer: 5

Hint: Expression Question
Question: Expand (x+1)^2.
Model response: The expansion is x^2 + 2x + 1.
Extracted answer: x^2 + 2x + 1

Hint: Extraction from Boxed
Question: Find the radius.
Model response: The radius is \\boxed{2\\sqrt{3}}.
Extracted answer: 2\\sqrt{3}
"""


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def find_boxed_content(s):
    results = []
    i = 0
    while True:
        start = s.find("\\boxed{", i)
        if start == -1:
            break
        brace_count = 1
        j = start + 7
        while j < len(s) and brace_count > 0:
            if s[j] == "{":
                brace_count += 1
            elif s[j] == "}":
                brace_count -= 1
            j += 1
        if brace_count == 0:
            results.append(s[start + 7 : j - 1])
            i = j
        else:
            break
    return results


def normalize_extracted_answer(extraction, choices):
    if extraction:
        extraction = str(extraction).strip()
        extraction = re.sub(
            r"^(The answer is|The correct option is|Option|Answer)\s*:?\s*",
            "", extraction, flags=re.IGNORECASE,
        )
        yes_no = re.match(r"^(Yes|No)(?:[,.]|\s).+", extraction, re.IGNORECASE)
        if yes_no:
            extraction = yes_no.group(1)
        if extraction.endswith(".") and len(extraction) < 10:
            extraction = extraction.rstrip(".")

    if not extraction:
        return ""

    if choices:
        m = re.search(r"^\s*\(?([A-D])\)?\s*$", extraction, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        if len(extraction) < 5:
            letters = re.findall(r"(?:^|\s|\()([A-D])(?:$|\)|\.|:)", extraction)
            if letters:
                return letters[0].upper()
        prefix = re.match(r"^\s*([A-D])[\.\)]\s+", extraction, re.IGNORECASE)
        if prefix:
            return prefix.group(1).upper()

    return extraction


def _create_test_prompt(query, response, is_multichoice):
    if is_multichoice:
        hint = (
            "Hint: This is a Multiple Choice Question. Extract only the option letter (A, B, C, D). "
            "If the model answer is a value (e.g., '5'), look at the options in the Question "
            "(e.g., 'A. 5') and return the corresponding letter (e.g., 'A')."
        )
    else:
        hint = "Hint: This is a Free-form Question. Extract the value or expression."
    return f"{DEMO_PROMPT.strip()}\n\n{hint}\nQuestion: {query}\n\nModel response: {response}\n\nExtracted answer: "


# ---------------------------------------------------------------------
# LLM Extractor
# ---------------------------------------------------------------------

class LLMExtractor:
    def __init__(self, api_key, base_url, model_name):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def extract(self, full_prompt):
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts answers from text."},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.0,
                max_tokens=100,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return ""


# ---------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------

def process_single_problem(problem, llm, smart):
    response = problem.get("response", "")
    text_input = problem.get("text_input", "")

    q_type = problem.get("question_type")
    if not q_type:
        is_mc = (
            "Options:" in text_input
            or "A." in text_input
            or re.match(r"^[A-E]$", str(problem.get("correct_answer", "")).strip())
        )
        q_type = "multiple-choice" if is_mc else "free-form"

    is_multichoice = q_type == "multiple-choice"
    choices = ["A", "B", "C", "D"] if is_multichoice else []

    # 1. Regex extraction
    extraction = ""
    boxed = find_boxed_content(response)
    if boxed:
        extraction = boxed[-1]

    if not extraction:
        patterns = [
            r"(?:The answer is|The final answer is)[:\s]*([A-D])\.",
            r"(?:The answer is|The final answer is)[:\s]*([A-D])\b",
            r"(?:The answer is|The final answer is)[:\s]*(\d+)",
            r"^\s*([A-D])\s*$",
        ]
        for pat in patterns:
            m = re.search(pat, response, re.MULTILINE | re.IGNORECASE)
            if m:
                extraction = m.group(1)
                break

    # 2. LLM extraction
    if llm and (not extraction or smart):
        query = text_input or problem.get("prompt", "")
        full_prompt = _create_test_prompt(query, response, is_multichoice)
        llm_result = llm.extract(full_prompt)
        if llm_result:
            extraction = llm_result

    normalized = normalize_extracted_answer(extraction, choices)

    res = problem.copy()
    res["extraction"] = extraction
    res["normalized_extraction"] = normalized
    return res


def run_extraction(
    input_file,
    output_dir=None,
    api_key=None,
    base_url=None,
    model=None,
    smart=False,
    no_llm=False,
    limit=None,
    num_workers=16,
):
    cfg = load_config()
    output_dir = output_dir or cfg.get("eval_output_dir", "./eval_results")
    api_key = api_key or cfg["api"].get("api_key") or get_api_key()
    base_url = base_url or cfg["api"].get("base_url", "") or get_base_url("")
    model = model or cfg["api"].get("default_model", get_default_model("qwen-plus"))

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {input_file}...")
    data = read_json(input_file)
    if limit and limit > 0:
        data = data[:limit]

    llm = None
    if not no_llm:
        if not api_key:
            raise ValueError(
                "Missing API key for extraction. Set DASHSCOPE_API_KEY "
                "(or GEOAUX_API_KEY), pass --api_key, or use --no_llm."
            )
        print(f"Initializing LLM extractor: {model} at {base_url}")
        llm = LLMExtractor(api_key, base_url, model)

    results = []
    print(f"Extracting answers with {num_workers} workers...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_problem, p, llm, smart): p for p in data
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing item: {e}")

    try:
        results.sort(key=lambda x: int(x["id"]) if str(x["id"]).isdigit() else str(x["id"]))
    except Exception:
        pass

    output_file = os.path.join(output_dir, "extracted_answers.json")
    save_json(results, output_file)
    print(f"Saved extracted results to {output_file}")
    return output_file
