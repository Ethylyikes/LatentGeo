# 🌍 LatentGeo

## 🗂️ Project Structure

```text
LatentGeo/
├── README.md
├── requirementsForRL.txt
├── requirementsForRLEval.txt
├── SFT/
│   ├── train.sh
│   ├── src/single_inference.py
│   └── data/example.json
├── RL/
│   └── jobs.sh
├── VLMEvalKit/
│   └── jobs.sh
└── Geoaux/
    ├── scripts/run_infer.py
    ├── scripts/run_eval.py
    └── data/example.json
```

## 🤗 Benchmark Dataset (Placeholder)

- HuggingFace link placeholder: `https://huggingface.co/datasets/<YOUR_ORG>/<YOUR_BENCHMARK_NAME>`
- Replace `<YOUR_ORG>/<YOUR_BENCHMARK_NAME>` with your real dataset path later.
- Default benchmark data paths:
  - `Geoaux/data/geoaux.json`
  - `Geoaux/data/image/`

## 🧪 SFT

### 1) Environment and Dependencies

```bash
cd /Users/ethyl/code/LatentGeo/SFT
conda create -n sft_env python=3.10 -y
conda activate sft_env

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install trl datasets accelerate qwen-vl-utils pillow

# Use local transformers
pip uninstall -y transformers
pip install -e ./transformers
```

### 2) Data Preparation

- Default training data path: `SFT/data/train.json`
- Example data: `SFT/data/example.json`
- If needed, update `DATA_PATH` in `SFT/train.sh`

Example format:

```json
{
  "image_input": "images/example_question_image.jpg",
  "image_output": "images/example_aux_image.jpg",
  "text_input": "<image>\nQuestion: ...",
  "aux_think": "Intermediate reasoning.",
  "text_output": "Final reasoning and answer.",
  "use_latent": true
}
```

### 3) Training

```bash
cd /Users/ethyl/code/LatentGeo/SFT
bash train.sh
```

### 4) Inference

- Edit `SFT/src/single_inference.py`:
  - `MODEL_PATH`
  - `IMAGE_PATH`
  - `TEXT_PROMPT`

```bash
cd /Users/ethyl/code/LatentGeo/SFT
python src/single_inference.py
```

## 🚀 RL Training

### 1) Environment and Dependencies

```bash
cd /Users/ethyl/code/LatentGeo
conda create -n rl_env python=3.10 -y
conda activate rl_env

pip install -r requirementsForRL.txt
```

### 2) Training

- Update `RL/jobs.sh` as needed
- Start training:

```bash
cd /Users/ethyl/code/LatentGeo/RL
bash jobs.sh
```

## 📊 RL Evaluation (VLMEvalKit)

### 1) Environment and Dependencies

```bash
cd /Users/ethyl/code/LatentGeo
conda create -n rl_eval_env python=3.10 -y
conda activate rl_eval_env

pip install -r requirementsForRLEval.txt
```

### 2) Configuration and Run

- Update `VLMEvalKit/jobs.sh` as needed
- For API-based evaluation, set your API key first (in shell or `.env`)

```bash
cd /Users/ethyl/code/LatentGeo/VLMEvalKit
bash jobs.sh
```

## 🧭 Geoaux Benchmark Inference

### 1) Environment and Dependencies

```bash
cd /Users/ethyl/code/LatentGeo/Geoaux
conda create -n geoaux_env python=3.10 -y
conda activate geoaux_env

pip install --upgrade pip
pip install torch transformers openai pyyaml tqdm pillow python-Levenshtein pandas
```

### 2) Inference

Local model:

```bash
cd /Users/ethyl/code/LatentGeo/Geoaux
python scripts/run_infer.py \
  --model_key qwen2.5-vl \
  --model_path Qwen/Qwen2.5-VL-7B-Instruct \
  --world_size 4
```

API model:

```bash
cd /Users/ethyl/code/LatentGeo/Geoaux
python scripts/run_infer.py \
  --model_key api \
  --model_path qwen-plus \
  --num_workers 16
```

Inference output:

- `Geoaux/output/<model_name>_results.json`

## ✅ Geoaux Benchmark Evaluation

### 1) Set Environment Variables (if using LLM extraction or judge)

```bash
export DASHSCOPE_API_KEY="your-api-key"
export DASHSCOPE_BASE_URL="your_url"
export DASHSCOPE_MODEL="qwen-plus"
```

### 2) Evaluation

End-to-end (extract + score):

```bash
cd /Users/ethyl/code/LatentGeo/Geoaux
python scripts/run_eval.py \
  --input_file ./output/model_results.json \
  --smart --use_judge
```

Extraction only:

```bash
cd /Users/ethyl/code/LatentGeo/Geoaux
python scripts/run_eval.py \
  --input_file ./output/model_results.json \
  --step extract --no_llm
```

Scoring only:

```bash
cd /Users/ethyl/code/LatentGeo/Geoaux
python scripts/run_eval.py \
  --input_file ./eval_results/extracted_answers.json \
  --step score --use_judge
```

Evaluation outputs:

- `Geoaux/eval_results/extracted_answers.json`
- `Geoaux/eval_results/scored_results.json`
- `Geoaux/eval_results/score_report.txt`

## 🙏 Acknowledgments

We thank the following excellent projects for their valuable contributions and inspiration:

- [GDPO](https://github.com/NVlabs/GDPO)
- [Monet](https://github.com/NOVAglow646/Monet)
- [Mirage](https://github.com/UMass-Embodied-AGI/Mirage)
- [MathCanvas](https://github.com/shiwk24/MathCanvas)
