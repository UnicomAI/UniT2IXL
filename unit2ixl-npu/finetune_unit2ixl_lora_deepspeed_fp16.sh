Network="UniT2IXLLoRADeepspeed"

model_name="UnicomAI/UniT2IXL"
train_data_dir=""
validation_prompt=""
batch_size=4
num_processors=4
max_train_steps=2000
checkpointing_steps=1000
validation_epochs=5
mixed_precision="fp16"
resolution=1024

for para in $*; do
  if [[ $para == --batch_size* ]]; then
    batch_size=$(echo ${para#*=})
  elif [[ $para == --max_train_steps* ]]; then
    max_train_steps=$(echo ${para#*=})
  elif [[ $para == --mixed_precision* ]]; then
    mixed_precision=$(echo ${para#*=})
  elif [[ $para == --resolution* ]]; then
    resolution=$(echo ${para#*=})
  elif [[ $para == --checkpointing_steps* ]]; then
    checkpointing_steps=$(echo ${para#*=})
  elif [[ $para == --validation_epochs* ]]; then
    validation_epochs=$(echo ${para#*=})
  fi
done

export TASK_QUEUE_ENABLE=2
export HCCL_CONNECT_TIMEOUT=1200
export ACLNN_CACHE_LIMIT=100000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
  test_path_dir=${cur_path}
  cd ..
  cur_path=$(pwd)
else
  test_path_dir=${cur_path}
fi

output_path=${cur_path}/output_lora


mkdir -p ${output_path}


start_time=$(date +%s)
echo "start_time: ${start_time}"

ASCEND_RT_VISIBLE_DEVICES="0,1,2,3" accelerate launch --config_file accelerate_deepspeed_config.yaml  \
  train_text_to_image_lora_unit2ixl.py \
  --pretrained_model_name_or_path=$model_name \
  --caption_column="text" \
  --resolution=$resolution  \
  --train_batch_size=$batch_size \
  --random_flip \
  --num_train_epochs=200 \
  --checkpointing_steps=$checkpointing_steps \
  --train_data_dir=$train_data_dir \
  --learning_rate=1e-05 \
  --validation_prompt=$validation_prompt \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision=$mixed_precision \
  --max_train_steps=$max_train_steps \
  --seed=42 \
  --noise_offset=0.1 \
  --enable_npu_flash_attention \
  --output_dir=${output_path} > ${output_path}/train_${mixed_precision}_unit2ixl_lora_deepspeed.log 2>&1 &
wait


#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

#结果打印，不需要修改
echo "------------------ Final result ------------------"

#输出性能FPS，需要模型审视修改
FPS=$(grep "FPS: " ${output_path}/train_${mixed_precision}_unit2ixl_lora_deepspeed.log | awk '{print $NF}' | sed -n '100,199p' | awk '{a+=$1}END{print a/NR}')

#获取性能数据，不需要修改
#吞吐量
ActualFPS=$(awk 'BEGIN{printf "%.2f\n", '${FPS}'}')

#打印，不需要修改
echo "Final Performance images/sec : $ActualFPS"

#loss值，不需要修改
ActualLoss=$(grep -o "step_loss=[0-9.]*" ${output_path}/train_${mixed_precision}_unit2ixl_lora_deepspeed.log | awk 'END {print $NF}')

#打印，不需要修改
echo "Final Train Loss : ${ActualLoss}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_'8p'_'acc'

#单迭代训练时长
TrainingTime=$(awk 'BEGIN{printf "%.2f\n", '${batch_size}'*'${num_processors}'/'${FPS}'}')

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${output_path}/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${output_path}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${output_path}/${CaseName}.log
echo "CaseName = ${CaseName}" >>${output_path}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>${output_path}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>${output_path}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>${output_path}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${output_path}/${CaseName}.log
