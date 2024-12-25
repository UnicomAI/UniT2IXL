export MODEL_NAME="UnicomAI/UniT2IXL"
export DATASET_NAME="lambdalabs/naruto-blip-captions"
export ACCELERATE_CONFIG_FILE="ac.yaml"
export CUDA_VISIBLE_DEVICES=0
accelerate launch --config_file $ACCELERATE_CONFIG_FILE  --main_process_port 12345 train_text_to_image_lora_unit2ixl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --caption_column="text" \
  --resolution=1024 --random_flip \
  --train_batch_size=4 \
  --num_train_epochs=200 --checkpointing_steps=1000 \
  --validation_epochs=10 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --rank=8 \
  --output_dir="sd-papercut-model-lora-sdxl" \
  --validation_prompt="一个龙的窗花" \
  --noise_offset=0.1 \
  --dataset_name=$DATASET_NAME