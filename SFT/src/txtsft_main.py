import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig, AutoProcessor
import os
import logging
from trl import SFTTrainer, SFTConfig
from qwen_vl_utils import process_vision_info
from utils import seed_everything, get_args, place_input_image, remove_assistant_images, generate_labels_after_multi_token_start, replace_visual_special_tokens, load_jsonl_dataset
from task import task_preprocess_config

# 设定随机种子
seed_everything(seed=55)

# 获取参数
args = get_args()

# 配置日志
local_rank = int(os.environ.get("LOCAL_RANK", -1))
handlers = [logging.StreamHandler()]
if local_rank in [-1, 0]:
    handlers.append(logging.FileHandler(args.log_file, mode='a', encoding='utf-8'))

logging.basicConfig(
    level=logging.INFO if local_rank in [-1, 0] else logging.WARN,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
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
    logging.warning(f"Failed to load processor from {args.model}. Assuming it's a checkpoint without processor files.")
    logging.warning("Trying to load processor from default 'Qwen/Qwen2.5-VL-7B-Instruct'.")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", cache_dir=cache_dir)

# 加载模型
model_path = args.model
config = Qwen2_5_VLConfig.from_pretrained(model_path, cache_dir=cache_dir)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, 
    config=config, 
    torch_dtype=torch.bfloat16, 
    cache_dir=cache_dir, 
    attn_implementation=args.attn_implementation
)

# 冻结视觉编码器 (与 main.py 保持一致，节省显存)
for param in model.visual.parameters():
    param.requires_grad = False

def collate_fn_sft(examples):
    # 移除助手回复中的图像
    user_examples = remove_assistant_images(examples)
    
    # 应用聊天模板
    texts = [processor.apply_chat_template(example, tokenize=False) for example in user_examples]
    
    # 替换输入图像占位符
    texts = [place_input_image(text) for text in texts]
    
    # 处理视觉信息
    image_inputs, _ = process_vision_info(user_examples)

    # Tokenize
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    
    # 生成 Labels
    pad_token_idx = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]
    
    # 遮盖 Prompt (包含输入图像) 和 Padding
    labels = generate_labels_after_multi_token_start(batch["input_ids"], answer_start_token_pattern, pad_token_idx)
    batch["labels"] = labels
    
    return batch

# 准备数据
preprocess_function = task_preprocess_config[args.task]
train_dataset = load_jsonl_dataset(args.data_path)
train_dataset = [preprocess_function(sample) for sample in train_dataset]

# 训练配置
training_args = SFTConfig(
    output_dir=args.save_model_path,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    dataloader_num_workers=16,
    dataloader_pin_memory=True,
    warmup_steps=10,
    learning_rate=1e-5,
    weight_decay=0.01,
    logging_steps=20,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,
    optim="adamw_torch_fused",
    bf16=True,
    push_to_hub=False,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
    ddp_find_unused_parameters=False,
    logging_dir='./logs/',
    logging_strategy='steps',
)

# 初始化 Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn_sft,
    tokenizer=processor.tokenizer,
)

# 开始训练
trainer.train()
trainer.save_model(training_args.output_dir)
