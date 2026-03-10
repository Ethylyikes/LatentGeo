import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
from qwen_vl_utils import process_vision_info
import argparse
import sys
import os

# Add src to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import place_input_image

def main():
    # Hardcoded paths and settings
    MODEL_PATH = "./checkpoints/stage3"
    IMAGE_PATH = "./data/images/Plane_Geometry/116330_question_0.png"
    TEXT_PROMPT = (
        "You are a helpful assistant. First, analyze the problem with image and describe any necessary auxiliary lines or visual reasoning steps. "
        "Then, provide a reasoning to solve the problem. Finally, give the final answer. Provide the final answer in the format: 'The final answer is: xxx.' .\n\n"
        "Question: \n"
        "In the figure, circle $O$ has radius $5$. Points $E$ and $F$ lie on the circle with $\\triangle EOF$ central angle $\\triangle EOF$ satisfying $\\nabla$ actually: $\\angle EOF=60^\\circ$. "
        "A regular (equilateral) triangle $ABC$ is inscribed so that vertices $A$ and $B$ lie on the radii $OE$ and $OF$ respectively, point $C$ lies on the arc $EF$, and $AB\\perp OF$. "
        "Find the side length of $\\triangle ABC$."
    )
    DEVICE = "cuda:0"
    LATENT_SIZE = 10

    print(f"Loading processor from {MODEL_PATH}...")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Failed to load processor from checkpoint ({e}). Using default Qwen processor.")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)

    # Add special tokens
    special_tokens = ["<|latent_pad|>", "<|latent_start|>", "<|latent_end|>"]
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = "<|endoftext|>"
    
    num_added = processor.tokenizer.add_tokens(special_tokens, special_tokens=True)
    if num_added > 0:
        print(f"Added {num_added} special tokens to tokenizer.")

    print(f"Loading model from {MODEL_PATH}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    
    # Ensure config has latent settings if not present
    if not hasattr(model.config, 'latent_size'):
        model.config.latent_size = LATENT_SIZE
    
    model.eval()

    # Prepare Image
    try:
        image = Image.open(IMAGE_PATH).convert("RGB")
        print(f"Loaded image: {IMAGE_PATH} ({image.size})")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Construct Prompt
    instruction = "<image>\n" + TEXT_PROMPT

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    # Preprocessing
    # 1. Apply chat template
    text_prompt = processor.apply_chat_template(conversation, tokenize=False)
    
    # 2. Fix image placement (remove auto-inserted image tokens and use <image> placeholder position)
    text_prompt = place_input_image(text_prompt, sep_token=None)
    
    # 3. Process vision info
    image_inputs, video_inputs = process_vision_info(conversation)

    # 4. Tokenize
    # Append assistant start token for generation
    prompt_with_assistant = text_prompt + '<|im_start|>assistant'
    
    print("Tokenizing input...")
    inputs = processor(
        text=[prompt_with_assistant],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(DEVICE)

    print("Generating...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            tokenizer=processor.tokenizer,
        )

    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.tokenizer.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=False, # Show special tokens like <|latent_start|>
        clean_up_tokenization_spaces=False
    )[0]

    print("\n" + "="*40)
    print("Inference Result (Raw Output):")
    print("="*40)
    print(output_text)
    print("="*40)

if __name__ == "__main__":
    main()
