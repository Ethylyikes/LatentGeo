# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, TypedDict
import re
import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto, DataProtoItem
from .config import RewardConfig, RuleBasedJudgeConfig
import numpy as np
import pdb
import inspect
class RewardScore(TypedDict, total=False):
    score: float
    overall: float
    format: Optional[float]
    accuracy: Optional[float]
    latent: Optional[float]
    length: Optional[float]


SequentialRewardFunction = Callable[[str, str], RewardScore]

BatchRewardFunction = Callable[[List[str], List[str]], List[RewardScore]]

SingleRuleBasedJudgeFunction = Callable[[str, str], RewardScore]

BatchRuleBasedJudgeFunction = Callable[[List[str], List[str]], List[RewardScore]]


def _latent_token_pair() -> tuple[str, str]:
    """Return (start, end) markers for latent blocks, allowing env overrides."""
    start_tok = os.environ.get("LATENT_START_TOKEN") or os.environ.get("ABS_VIS_START_TOKEN") or "<abs_vis_token>"
    end_tok = os.environ.get("LATENT_END_TOKEN") or os.environ.get("ABS_VIS_END_TOKEN") or "</abs_vis_token>"
    return start_tok, end_tok


def replace_abs_vis_token_content(s: str) -> str:
    start_tok, end_tok = _latent_token_pair()
    try:
        pattern = re.compile(rf'({re.escape(start_tok)})(.*?)({re.escape(end_tok)})', flags=re.DOTALL)
    except re.error:
        return s
    return pattern.sub(r'\1<latent>\3', s)


def _extract_total_reward(score: Dict[str, float]) -> Optional[float]:
    """Prefer `score`, fall back to legacy `overall`."""
    for key in ("score", "overall"):
        if key in score and isinstance(score[key], (float, int, np.floating, np.integer)):
            return float(score[key])
    return None


class FunctionRewardManager(ABC):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer

    @abstractmethod
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """Compute reward for a batch of data."""
        ...


class SequentialFunctionRewardManager(FunctionRewardManager):
    reward_fn: SequentialRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        sig = None
        try:
            sig = inspect.signature(self.reward_fn)
        except Exception:
            pass
        for i in range(len(data)):
            valid_response_ids = response_ids[i][: response_length[i]]
            # Preserve latent markers for compute_score_wzh-style rewards to allow latent detection.
            if self.config.reward_function_name == "compute_score_wzh":
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
            else:
                response_str = self.tokenizer.decode(
                    valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
                )
            ground_truth = data.non_tensor_batch["ground_truth"][i]

            answer_label = None
            if "answer_label" in data.non_tensor_batch:
                try:
                    answer_label = data.non_tensor_batch["answer_label"][i]
                except Exception:
                    answer_label = None

            try:
                if sig is not None and "answer_label" in sig.parameters:
                    score = self.reward_fn(response_str, ground_truth, answer_label=answer_label)
                else:
                    score = self.reward_fn(response_str, ground_truth)
            except Exception:
                score = self.reward_fn(response_str, ground_truth)
            total_reward = _extract_total_reward(score)
            if total_reward is not None:
                reward_tensor[i, response_length[i] - 1] = total_reward
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        response_str, ground_truth = [], []
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        # Gracefully fall back when ref_resp_lengths is missing (e.g., validation batches).
        ref_resp_lengths = data.non_tensor_batch.get("ref_resp_lengths", response_length)
        for i in range(len(data)):
            valid_response_ids = response_ids[i][: response_length[i]]
            if "latent_reward_function" in (self.config.reward_function or ""):
                response_str_=replace_abs_vis_token_content(self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)).replace("<|endoftext|>", "").replace("<|im_end|>", "")
            elif self.config.reward_function_name == "compute_score_wzh":
                response_str_=self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
            else:
                response_str_=self.tokenizer.decode(valid_response_ids, skip_special_tokens=self.config.skip_special_tokens)
            response_str.append(response_str_)
            ground_truth.append(data.non_tensor_batch["ground_truth"][i])

        #breakpoint()
        extra_kwargs = {}
        progress = None
        phase = None
        global_step_val = None
        training_steps_val = None
        try:
            if "progress" in data.non_tensor_batch:
                progress = float(data.non_tensor_batch["progress"][0])
            if "phase" in data.non_tensor_batch:
                phase = int(data.non_tensor_batch["phase"][0])
            if "global_step" in data.non_tensor_batch:
                global_step_val = int(data.non_tensor_batch["global_step"][0])
            if "training_steps" in data.non_tensor_batch:
                training_steps_val = int(data.non_tensor_batch["training_steps"][0])
        except Exception:
            pass

        try:
            sig = inspect.signature(self.reward_fn)
            if "length_penalty_weight" in sig.parameters:
                extra_kwargs["length_penalty_weight"] = self.config.length_penalty_weight
            if "labels" in sig.parameters and "label" in data.non_tensor_batch:
                labels = data.non_tensor_batch["label"]
                try:
                    labels = labels.tolist()
                except Exception:
                    pass
                extra_kwargs["labels"] = labels
            if "answer_labels" in sig.parameters and "answer_label" in data.non_tensor_batch:
                answer_labels = data.non_tensor_batch["answer_label"]
                try:
                    answer_labels = answer_labels.tolist()
                except Exception:
                    pass
                extra_kwargs["answer_labels"] = answer_labels
            if "tokenizer" in sig.parameters:
                extra_kwargs["tokenizer"] = self.tokenizer
            if "progress" in sig.parameters and progress is not None:
                extra_kwargs["progress"] = progress
            if "phase" in sig.parameters and phase is not None:
                extra_kwargs["phase"] = phase
            if "global_step" in sig.parameters and global_step_val is not None:
                extra_kwargs["global_step"] = global_step_val
            if "training_steps" in sig.parameters and training_steps_val is not None:
                extra_kwargs["training_steps"] = training_steps_val
        except Exception:
            pass

        if "single_step_rewards" in data.non_tensor_batch:  # mc List[float]
            scores = self.reward_fn(response_str, data.non_tensor_batch["single_step_rewards"])
        elif "full_step_rewards" in data.non_tensor_batch:  # mc2 List[List[float]]
            scores = self.reward_fn(
            response_str,
            data.non_tensor_batch["full_step_rewards"],
            resp_lengths=response_length,
            ref_resp_lengths=ref_resp_lengths,
            **extra_kwargs,
            )
        elif "correctness" in data.non_tensor_batch:
            #pdb.set_trace()
            scores = self.reward_fn(
            response_str,
            data.non_tensor_batch["correctness"],
            resp_lengths=response_length,
            ref_resp_lengths=ref_resp_lengths,
            **extra_kwargs,
            )
        else:
            scores = self.reward_fn(
            response_str,
            ground_truth,
            resp_lengths=response_length,
            ref_resp_lengths=ref_resp_lengths,
            **extra_kwargs,
            )
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            total_reward = _extract_total_reward(score)
            if total_reward is not None:
                reward_tensor[i, response_length[i] - 1] = total_reward
            elif "score_step_wise" in score:
                poss = data.non_tensor_batch["step_end_positions"][i]
                reward_tensor[i, poss] = torch.tensor(score["score_step_wise"], dtype=reward_tensor.dtype)
            elif "overall_step_wise" in score: # mc2
                poss = data.non_tensor_batch["step_end_positions"][i]
                reward_tensor[i, poss] = torch.tensor(score["overall_step_wise"], dtype=reward_tensor.dtype)

            for key, value in score.items():
                if not (isinstance(value, np.floating) or isinstance(value, float)):
                    continue
                reward_metrics[key].append(value)
        #breakpoint()
        return reward_tensor, reward_metrics







class FunctionRuleBasedJudgeManager(ABC):
    """RuleBasedJudge manager for rule-based rule_based_judge."""

    def __init__(self, config: RuleBasedJudgeConfig, tokenizer: PreTrainedTokenizer):
        if config.judge_function is None:
            raise ValueError("RuleBasedJudge function is not provided.")

        if not os.path.exists(config.judge_function):
            raise FileNotFoundError(f"RuleBasedJudge function file {config.judge_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_rule_based_judge_fn", config.judge_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_rule_based_judge_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load rule_based_judge function: {e}")

        if not hasattr(module, config.judge_function_name):
            raise AttributeError(f"Module {module} does not have function {config.judge_function_name}.")

        rule_based_judge_fn = getattr(module, config.judge_function_name)
        print(f"Using rule_based_judge function `{config.judge_function_name}` from `{config.judge_function}`.")
        self.rule_based_judge_fn = rule_based_judge_fn
        self.config = config
        self.tokenizer = tokenizer

    @abstractmethod
    def compute_rule_based_judge(self, data: DataProto) -> bool:
        """Compute rule_based_judge for a batch of data."""
        ...
    
    def compute_rule_based_judge_with_string(self, response_str: str, ground_truth: str) -> bool:
        """Compute rule_based_judge for a single response string."""
        ...


class SingleFunctionRuleBasedJudgeManager(FunctionRuleBasedJudgeManager):
    rule_based_judge_fn: SingleRuleBasedJudgeFunction

    def compute_rule_based_judge(self, data: DataProtoItem) -> bool:
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)

        valid_response_ids = response_ids[: response_length]
        response_str = self.tokenizer.decode(
            valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
        )
        ground_truth = data.non_tensor_batch["ground_truth"]
        try:
            correctness = self.rule_based_judge_fn(response_str, ground_truth)
        except Exception as e:
            print(f"Rule-based judge error: {e}")
            correctness = False
        return correctness, response_str

    def compute_rule_based_judge_with_string(self, response_str: str, ground_truth: str) -> bool:
        try:
            correctness = self.rule_based_judge_fn(response_str, ground_truth)
        except Exception as e:
            print(f"Rule-based judge error: {e}")
            correctness = False
        return correctness

class BatchFunctionRuleBasedJudgeManager(FunctionRuleBasedJudgeManager):
    rule_based_judge_fn: BatchRuleBasedJudgeFunction

    def compute_rule_based_judge(self, data: DataProto) -> List[bool]:
        correctness = []
        response_strs = []
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        for i in range(len(data)):
            valid_response_ids = response_ids[i][: response_length[i]]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=self.config.skip_special_tokens)
            response_strs.append(response_str)
            ground_truth = data.non_tensor_batch["ground_truth"][i]
            try:
                correctness = self.rule_based_judge_fn(response_str, ground_truth)
            except Exception as e:
                print(f"Rule-based judge error: {e}")
                correctness = False
            correctness.append(correctness)
        return correctness
