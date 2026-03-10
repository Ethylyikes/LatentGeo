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

from typing import Optional

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, collate_fn
from .config import DataConfig


def create_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]) -> None:
    def _apply_max_samples(dataset, max_samples: Optional[int], split: str):
        """Normalize the `*_max_samples` knob.

        Treat negative values as "use full dataset", raise on zeros, and clamp
        to the dataset length when a positive cap is provided.
        """
        if max_samples is None or max_samples < 0:
            return dataset
        if max_samples == 0:
            raise ValueError(f"{split}_max_samples is 0; this would create an empty {split} dataset.")
        capped = min(max_samples, len(dataset))
        if capped < len(dataset):
            dataset = torch.utils.data.Subset(dataset, list(range(capped)))
        return dataset

    def _guard_non_empty(dataset, path: str, split: str):
        if len(dataset) == 0:
            raise ValueError(
                f"{split.capitalize()} dataset is empty after loading {path}. "
                "Please check the path, filtering rules, and prompt/image fields."
            )

    train_dataset = RLHFDataset(
        data_path=config.train_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_and_invalid_prompts=config.filter_overlong_and_invalid_prompts,
    )

    train_dataset = _apply_max_samples(train_dataset, config.train_max_samples, "train")
    _guard_non_empty(train_dataset, config.train_files, "train")
    train_dataset_len = len(train_dataset)
    
    # use sampler for better ckpt resume
    if config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    train_batch_size = config.rollout_batch_size
    train_drop_last = True
    if train_dataset_len < train_batch_size:
        # Keep at least one batch when the custom dataset is small.
        train_batch_size = train_dataset_len
        train_drop_last = False
        print(
            f"[DataLoader] train dataset has only {train_dataset_len} samples; "
            f"lowering batch_size from {config.rollout_batch_size} to {train_batch_size} "
            "and disabling drop_last to avoid an empty dataloader."
        )

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=getattr(config, "dataloader_num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=train_drop_last,
    )

    val_dataset = RLHFDataset(
        data_path=config.val_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_and_invalid_prompts=config.filter_overlong_and_invalid_prompts,
    )
    
    val_dataset = _apply_max_samples(val_dataset, config.val_max_samples, "val")
    _guard_non_empty(val_dataset, config.val_files, "val")
    val_dataset_len = len(val_dataset)
    
    val_batch_size = len(val_dataset) if config.val_batch_size == -1 else config.val_batch_size

    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=getattr(config, "dataloader_num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    assert len(train_dataloader) >= 1
    assert len(val_dataloader) >= 1
    print(
        f"Train samples: {train_dataset_len}, "
        f"batch_size: {train_batch_size}, "
        f"train dataloader steps/epoch: {len(train_dataloader)}"
    )
    print(
        f"Val samples: {val_dataset_len}, "
        f"batch_size: {val_batch_size}, "
        f"val dataloader steps/epoch: {len(val_dataloader)}"
    )
    return train_dataloader, val_dataloader
