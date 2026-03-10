import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig, AutoTokenizer, AutoProcessor
from PIL import Image
import os
import logging

from trl import SFTTrainer, SFTConfig
from qwen_vl_utils import process_vision_info

from utils import *
from task import *
from trainer import CustomTrainerStage1, CustomTrainerStage2, CustomTrainerStage3

# 设定随机种子以确保结果可复现
seed_everything(seed=55)
# 获取命令行参数
args=get_args()

# 配置日志记录
local_rank = int(os.environ.get("LOCAL_RANK", -1))
handlers = [logging.StreamHandler()]
if local_rank in [-1, 0]:
    handlers.append(logging.FileHandler(args.log_file, mode='a', encoding='utf-8'))

logging.basicConfig(
    level=logging.INFO if local_rank in [-1, 0] else logging.WARN,  # 日志级别
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    datefmt='%Y-%m-%d %H:%M:%S',  # 日期格式
    handlers=handlers,
)

logging.info('=='*20)
logging.info(args)
logging.info('=='*20)

# 加载模型和处理器
cache_dir = args.cache_dir
os.environ['HF_HOME'] = cache_dir


try:
    processor = AutoProcessor.from_pretrained(args.model, cache_dir=cache_dir)
except (OSError, ValueError):
    # 如果 args.model 是 checkpoint 路径且缺少 processor 文件，尝试回退
    logging.warning(f"Failed to load processor from {args.model}. Assuming it's a checkpoint without processor files.")
    logging.warning("Trying to load processor from default 'Qwen/Qwen2.5-VL-7B-Instruct'.")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", cache_dir=cache_dir)

# 向 tokenizer 中添加用于潜空间表征的特殊 token
processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)


# 根据训练阶段加载模型和配置
if args.stage == 'stage1': 
    model_path = args.model
    config = Qwen2_5_VLConfig.from_pretrained(model_path, cache_dir=cache_dir)
    grad_checkpointing = True 
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, cache_dir=cache_dir, attn_implementation=args.attn_implementation)
    model.resize_token_embeddings(len(processor.tokenizer))

elif args.stage == 'stage2':
    # Stage 2: 类似于 Stage 3，我们可能希望加载 Stage 1 的 checkpoint
    if args.load_model_path:
        logging.info(f"Stage 2: Loading model from checkpoint {args.load_model_path}")
        model_path = args.load_model_path
        # 注意：从 checkpoint 加载时，config 应该已经在 checkpoint 中
        config = Qwen2_5_VLConfig.from_pretrained(model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, attn_implementation=args.attn_implementation)
        # Stage 1 checkpoint 已经 resize 过了，所以这里通常不需要再 resize，
        # 除非 checkpoint 保存时没有保存 resize 后的 config/vocab。
        # 通常 trainer.save_model 会保存完整的。
    else:
        # 如果没有指定 load_model_path，则像 Stage 1 一样从 base model 开始
        logging.info(f"Stage 2: Loading model from base {args.model}")
        model_path = args.model
        config = Qwen2_5_VLConfig.from_pretrained(model_path, cache_dir=cache_dir)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, cache_dir=cache_dir, attn_implementation=args.attn_implementation)
        model.resize_token_embeddings(len(processor.tokenizer))
    
    grad_checkpointing = True

elif args.stage == 'stage3':
    model_path = args.load_model_path
    config = Qwen2_5_VLConfig.from_pretrained(model_path)
    grad_checkpointing = True
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, attn_implementation=args.attn_implementation)

# 更新模型配置
config.compress_strategy = args.compress_strategy
config.latent_size = args.latent_size
config.stage = args.stage
config.output_hidden_states = True
config.return_dict = True
try:
    model.model.config.compress_strategy = args.compress_strategy
    model.model.config.latent_size = args.latent_size
except Exception:
    pass

# 获取潜空间相关 token 的 ID
latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]
# 将 token ID 保存到模型配置中
model.config.latent_token_id = int(latent_token_idx)
model.config.latent_start_id = int(latent_start_idx)
model.config.latent_end_id = int(latent_end_idx)

# 冻结视觉编码器的参数，训练时只更新语言模型部分
for param in model.visual.parameters():
    param.requires_grad = False


# 第一阶段的数据整理函数
def collate_fn_stage1(examples):
    # 应用聊天模板
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]

    # 替换文本中的图像占位符
    texts = [place_input_image(text) for text in texts]
    texts = [place_output_image(text) for text in texts]
    texts = replace_visual_special_tokens(texts)

    # 处理视觉信息
    image_inputs, _ = process_vision_info(examples)

    # 分别处理用户和 assistant 的图像和文本
    user_examples = remove_assistant_images(examples)
    user_text = [processor.apply_chat_template(example, tokenize=False) for example in user_examples]
    user_text = replace_visual_special_tokens(user_text)
    user_image_inputs, _ = process_vision_info(user_examples)
    user_batch = processor(text=user_text, images=user_image_inputs, return_tensors="pt", padding=True)

    assistant_examples = remove_user_images(examples)
    assistant_text = [processor.apply_chat_template(example, tokenize=False) for example in assistant_examples]
    assistant_text = replace_visual_special_tokens(assistant_text)
    assistant_image_inputs, _ = process_vision_info(assistant_examples)
    assistant_batch = processor(text=assistant_text, images=assistant_image_inputs, return_tensors="pt", padding=True)

    # 创建最终的 batch
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    # 将用户图像数据放入 batch
    batch['pixel_values'] = user_batch.get('pixel_values')
    batch['image_grid_thw'] = user_batch.get('image_grid_thw')

    # 将 assistant 图像数据作为潜空间表征的目标
    # 注意：如果样本中没有 image_output (use_latent=False)，assistant_batch['pixel_values'] 可能为空
    batch['pixel_values_latent'] = assistant_batch.get('pixel_values')
    batch['image_grid_thw_latent'] = assistant_batch.get('image_grid_thw')

    # 获取特殊 token 的 ID
    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
    latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]
    pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]

    # 处理 batch，插入潜空间 token
    new_input_ids, new_attention_mask = process_batch(batch["input_ids"], batch["attention_mask"], 
                                                      latent_start_idx, latent_end_idx, latent_token_idx, args.latent_size, pad_token_idx)

    batch["input_ids"] = new_input_ids
    batch["attention_mask"] = new_attention_mask

    # 生成用于计算损失的 labels
    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]
    labels = generate_labels_after_multi_token_start(batch["input_ids"], answer_start_token_pattern, pad_token_idx, latent_token_idx)
    batch["labels"] = labels

    # 创建一个 mask，用于在计算损失时忽略图像输出部分的 token
    image_out_mask = mask_image_output_tokens(batch["input_ids"], latent_start_idx, latent_token_idx)
    batch["image_out_mask"] = image_out_mask

    return batch

# Stage 2 (Bridge) 数据整理函数
def collate_fn_stage2(examples):
    # Unpack examples if they are dicts with 'conversations'
    real_examples = []
    for ex in examples:
        if isinstance(ex, dict) and "conversations" in ex:
            if "image_path" in ex:
                logging.info(f"Processing sample with image: {ex['image_path']}")
            real_examples.append(ex["conversations"])
        else:
            real_examples.append(ex)
    examples = real_examples

    # --- 1. Image 分支 (同 Stage 1) ---
    texts_img = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    texts_img = [place_input_image(text) for text in texts_img]
    texts_img = [place_output_image(text) for text in texts_img]
    texts_img = replace_visual_special_tokens(texts_img)
    
    image_inputs_img, _ = process_vision_info(examples)
    
    # 我们需要分别为用户和助手处理图像，以获取 pixel_values_latent
    # (复制 Stage 1 的逻辑以保持一致性)
    user_examples = remove_assistant_images(examples)
    user_image_inputs, _ = process_vision_info(user_examples) # 用于 pixel_values
    user_text = [processor.apply_chat_template(example, tokenize=False) for example in user_examples]
    user_text = replace_visual_special_tokens(user_text)
    user_batch = processor(text=user_text, images=user_image_inputs, return_tensors="pt", padding=True)
    
    assistant_examples = remove_user_images(examples)
    assistant_image_inputs, _ = process_vision_info(assistant_examples) # 用于 pixel_values_latent
    assistant_text = [processor.apply_chat_template(example, tokenize=False) for example in assistant_examples]
    assistant_text = replace_visual_special_tokens(assistant_text)
    assistant_batch = processor(text=assistant_text, images=assistant_image_inputs, return_tensors="pt", padding=True)
    
    # 为 Image 分支创建 Batch
    batch_img = processor(text=texts_img, images=image_inputs_img, return_tensors="pt", padding=True)
    
    # --- 2. Text 分支 (移除输入图像) ---
    txt_examples = remove_user_images(examples) # 从用户回合中移除输入图像
    
    texts_txt = [processor.apply_chat_template(example, tokenize=False) for example in txt_examples]
    # 移除 <image> 占位符，避免插入图像 token
    texts_txt = [text.replace('<image>', '') for text in texts_txt] 
    texts_txt = [place_input_image(text) for text in texts_txt] # 如果 <image> 已移除，这步应该无操作
    texts_txt = [place_output_image(text) for text in texts_txt] # 保留输出图像 (latent) 的占位符
    texts_txt = replace_visual_special_tokens(texts_txt)
    
    # 注意: process_vision_info 可能会返回助手图像 (latent 目标) 的信息
    # 但对于输入序列 (User)，没有图像。
    # 尽管我们移除了输入图像，但保留了输出图像 (latent)。
    # 这里的 image_inputs_txt 应该只包含输出图像。
    image_inputs_txt, _ = process_vision_info(txt_examples) 
    
    batch_txt = processor(text=texts_txt, images=image_inputs_txt, return_tensors="pt", padding=True)
    
    # --- 3. 后处理与合并 ---
    
    # 辅助函数：处理 batch (插入 latent token, 生成 label 等)
    def post_process_batch(b):
        # 插入 pixel values (输入图像)
        if 'pixel_values' not in b:
            # 对于 Text 分支，这里可能为空，后续会处理
            pass
            
        latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
        latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
        latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]
        pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]

        new_input_ids, new_attention_mask = process_batch(b["input_ids"], b["attention_mask"], 
                                                          latent_start_idx, latent_end_idx, latent_token_idx, args.latent_size, pad_token_idx)
        b["input_ids"] = new_input_ids
        b["attention_mask"] = new_attention_mask
        
        answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]
        labels = generate_labels_after_multi_token_start(b["input_ids"], answer_start_token_pattern, pad_token_idx, latent_token_idx)
        b["labels"] = labels
        
        image_out_mask = mask_image_output_tokens(b["input_ids"], latent_start_idx, latent_token_idx)
        b["image_out_mask"] = image_out_mask
        return b

    # 先独立处理两个 batch (处理 token 插入/padding)
    # 我们必须在 post_process 之前覆盖 pixel_values 逻辑，以模仿 Stage 1 的行为
    # Stage 1: batch['pixel_values'] 来自 USER batch (仅输入图像)。
    # batch['pixel_values_latent'] 来自 ASSISTANT batch。
    
    # 修正 Image Batch
    batch_img['pixel_values'] = user_batch['pixel_values'] # 输入图像
    batch_img['image_grid_thw'] = user_batch['image_grid_thw']
    batch_img['pixel_values_latent'] = assistant_batch['pixel_values']
    batch_img['image_grid_thw_latent'] = assistant_batch['image_grid_thw']
    # 修正 Text Batch
    # 输入图像: 无。
    # 对于 Text 分支，没有输入图像，但仍需要保证格式一致性
    # 创建一个空的 pixel_values 张量，避免 None 导致的下游处理问题
    batch_txt_num_samples = batch_txt['input_ids'].shape[0]
    # 使用空张量而非 None，确保模型 forward pass 不出错
    batch_txt['pixel_values'] = torch.zeros((batch_txt_num_samples, 0, 4096), dtype=torch.bfloat16, device=batch_txt['input_ids'].device)
    batch_txt['image_grid_thw'] = None
    batch_txt['pixel_values_latent'] = assistant_batch['pixel_values'] # 目标相同
    batch_txt['image_grid_thw_latent'] = assistant_batch['image_grid_thw'] # 目标相同
    batch_txt = post_process_batch(batch_txt)
    
    # 拼接 (Concatenate)
    # 我们需要将 input_ids/labels/attention_mask padding 到两个 batch 的最大长度
    max_len = max(batch_img['input_ids'].shape[1], batch_txt['input_ids'].shape[1])
    
    def pad_tensor(t, length, pad_val):
        # t: [B, L]
        if t.shape[1] >= length: return t
        pad = torch.full((t.shape[0], length - t.shape[1]), pad_val, dtype=t.dtype, device=t.device)
        return torch.cat([t, pad], dim=1)
        
    pad_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
    
    final_input_ids = torch.cat([
        pad_tensor(batch_img['input_ids'], max_len, pad_token_id),
        pad_tensor(batch_txt['input_ids'], max_len, pad_token_id)
    ], dim=0)
    
    final_attention_mask = torch.cat([
        pad_tensor(batch_img['attention_mask'], max_len, 0),
        pad_tensor(batch_txt['attention_mask'], max_len, 0)
    ], dim=0)
    
    final_labels = torch.cat([
        pad_tensor(batch_img['labels'], max_len, -100),
        pad_tensor(batch_txt['labels'], max_len, -100)
    ], dim=0)
    
    final_image_out_mask = torch.cat([
        pad_tensor(batch_img['image_out_mask'], max_len, 0),
        pad_tensor(batch_txt['image_out_mask'], max_len, 0)
    ], dim=0)
    
    final_batch = {
        "input_ids": final_input_ids,
        "attention_mask": final_attention_mask,
        "labels": final_labels,
        "image_out_mask": final_image_out_mask,
        "pixel_values": batch_img['pixel_values'], # 仅前半部分有值
        "image_grid_thw": batch_img['image_grid_thw'], # 仅前半部分有值
        # 拼接 latent pixel values (两部分都有目标)
        "pixel_values_latent": torch.cat([batch_img['pixel_values_latent'], batch_txt['pixel_values_latent']], dim=0),
        "image_grid_thw_latent": torch.cat([batch_img['image_grid_thw_latent'], batch_txt['image_grid_thw_latent']], dim=0)
    }
    
    # 添加调试日志
    if local_rank in [-1, 0]:  # 仅在主进程输出日志
        logging.debug(f"[Stage 2 collate_fn] Image branch: input_ids {batch_img['input_ids'].shape}, pixel_values {batch_img['pixel_values'].shape if batch_img['pixel_values'] is not None else 'None'}")
        logging.debug(f"[Stage 2 collate_fn] Text branch: input_ids {batch_txt['input_ids'].shape}, pixel_values {batch_txt['pixel_values'].shape}")
        logging.debug(f"[Stage 2 collate_fn] Final batch: input_ids {final_input_ids.shape}, image_out_mask {final_image_out_mask.shape}")
        logging.debug(f"[Stage 2 collate_fn] pixel_values_latent {final_batch['pixel_values_latent'].shape}")
    
    return final_batch


# 根据任务类型选择预处理函数
preprocess_function = task_preprocess_config[args.task]
# 加载并预处理数据集
train_dataset = load_jsonl_dataset(args.data_path)
train_dataset = [preprocess_function(sample) for sample in train_dataset]


# Stage 3 数据整理函数
def collate_fn_stage3(examples):
    # 应用聊天模板
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]

    # 替换文本中的图像占位符
    texts = [place_input_image(text) for text in texts]
    texts = [place_output_image(text) for text in texts]
    texts = replace_visual_special_tokens(texts)

    # 处理视觉信息 (仅输入图像)
    image_inputs, _ = process_vision_info(examples)

    # 创建 batch
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    # 获取特殊 token 的 ID
    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
    latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]
    pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]

    # 处理 batch，插入潜空间 token
    new_input_ids, new_attention_mask = process_batch(batch["input_ids"], batch["attention_mask"], 
                                                      latent_start_idx, latent_end_idx, latent_token_idx, args.latent_size, pad_token_idx)

    batch["input_ids"] = new_input_ids
    batch["attention_mask"] = new_attention_mask

    # 生成用于计算损失的 labels
    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]
    labels = generate_labels_after_multi_token_start(batch["input_ids"], answer_start_token_pattern, pad_token_idx, latent_token_idx)
    batch["labels"] = labels
    
    return batch


# 根据训练阶段选择自定义的 Trainer 和数据整理函数
if args.stage in ['stage1']:
    CustomTrainer = CustomTrainerStage1
    collate_fn = collate_fn_stage1
elif args.stage in ['stage2']:
    CustomTrainer = CustomTrainerStage2
    collate_fn = collate_fn_stage2
else:
    CustomTrainer = CustomTrainerStage3
    collate_fn = collate_fn_stage3

# 配置 SFTTrainer 的训练参数
training_args = SFTConfig(
    output_dir=args.save_model_path,
    num_train_epochs=args.epochs,
    max_seq_length=args.max_seq_length,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    # --- 新增这行 ---
    dataloader_num_workers=16,  # 配合你申请的 32 核 CPU，建议设为 16 或 24
    dataloader_pin_memory=True, # 建议开启，加速 CPU -> GPU 传输
    
    warmup_steps=10,
    learning_rate=args.learning_rate,
    weight_decay=0.01,
    logging_steps=20,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,
    optim="adamw_torch_fused",
    bf16=True,
    push_to_hub=False,
    remove_unused_columns=False,
    gradient_checkpointing=grad_checkpointing,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
    ddp_find_unused_parameters=False,
    logging_dir='./logs/',
    logging_strategy='steps',
)

# 初始化 Trainer
if args.stage == 'stage1':
    trainer = CustomTrainerStage1(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn_stage1,
        tokenizer=processor.tokenizer,
    )
elif args.stage == 'stage2':
    trainer = CustomTrainerStage2(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn_stage2,
        tokenizer=processor.tokenizer,
    )
elif args.stage == 'stage3':
        trainer = CustomTrainerStage3(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collate_fn_stage3,
            tokenizer=processor.tokenizer,
        )

# 开始训练
trainer.train()
# 保存模型
trainer.save_model(training_args.output_dir)
