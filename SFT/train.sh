export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA_PATH=./data/train.json
MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct

# stage1
torchrun --nproc_per_node=4 src/main.py \
    --model ${MODEL_NAME} \
    --epochs 5 \
    --task latentgeo \
    --latent_size 10 \
    --stage stage1 \
    --data_path ${DATA_PATH} \
    --log_file ./log.txt \
    --save_model_path ./checkpoints/stage1

# stage2
torchrun --nproc_per_node=4 src/main.py \
    --model ${MODEL_NAME} \
    --epochs 2 \
    --task latentgeo \
    --latent_size 10 \
    --stage stage2 \
    --data_path ${DATA_PATH} \
    --log_file ./log.txt \
    --load_model_path ./checkpoints/stage1 \
    --save_model_path ./checkpoints/stage2

# stage3
torchrun --nproc_per_node=4 src/main.py \
    --model ${MODEL_NAME} \
    --epochs 3 \
    --task latentgeo \
    --latent_size 10 \
    --stage stage3 \
    --data_path ${DATA_PATH} \
    --log_file ./log.txt \
    --load_model_path ./checkpoints/stage2 \
    --save_model_path ./checkpoints/stage3
