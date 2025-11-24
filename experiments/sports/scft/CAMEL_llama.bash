dataset="sports"
lora_rank=8
lora_trainable="q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj"
modules_to_save="null"
lora_dropout=0.1
LR=2e-4
model_choice="LlamaVision"
model_name_or_path="/hpc2hdd/home/zrao690/Llama-3.2-11B-Vision-Instruct"    # LLM底座模型路径，或者是huggingface hub上的模型名称
your_data_path="data/${dataset}/handled"  # 填入数据集所在的文件夹路径
your_checkpopint_path="saved"  # 填入用来存储模型的路径
MAX_STEPS=4000
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
_date="0730"
prefix=""
MAX_SOURCE_LENGTH=1024
MULTIMODAL_EMB="True"
POOLING="concat"
TRAIN_FILE=""
NUM_MOE=8
fine_tune="moelora"
gate="task_id"
freeze_visual_tower="True"
output_file=""
info_nce="False"

if [ "${MULTIMODAL_EMB}" == "True" ]; then
    TRAIN_FILE="multimodal_item_str.jsonline"
    prefix="${model_choice}_${fine_tune}_moe${NUM_MOE}_mean_${_date}"
else
    TRAIN_FILE="multimodal_item_str.jsonline"
    prefix="${model_choice}_text_${fine_tune}_${POOLING}_${_date}"
fi

if [ "${freeze_visual_tower}" == "True" ]; then
    freeze_visual_tower="True"
else
    prefix="${prefix}_unfreeze_visual"
fi

if [ "${info_nce}" == "True" ]; then
    prefix="${prefix}_infonce"
fi


if [ "${fine_tune}" == "moelora" ]; then
    enable_moelora="True"
else
    enable_moelora="False"
fi

if [ "${POOLING}" == "concat" ]; then
    concat_features="True"
    output_file="${prefix}_concat"
else
    concat_features="False"
    output_file="${prefix}"
fi




if [ "${gate}" == "task_id" ]; then
    enable_task_gate="True"
else
    enable_task_gate="False"
fi



peft_path=""  # 如果之前训练过，且存储了peft权重，则设置为peft权重的文件夹路径
export CUDA_VISIBLE_DEVICES=0
export NCCL_IB_DISABLE="1" 
export NCCL_P2P_DISABLE="1" 
#  --resume_from_checkpoint "/hpc2hdd/home/zrao690/MultimodalEmb/saved/beauty/moelora-mm_0323_avg/checkpoint-4000"\
# Training Command
# deepspeed --master_port $MASTER_PORT main_llm.py \
#     --deepspeed llm/ds_stage2.config \
#     --do_train \
#     --multimodal_embed $MULTIMODAL_EMB \
#     --enable_moelora $enable_moelora \
#     --num_moe $NUM_MOE \
#     --enable_task_gate $enable_task_gate \
#     --info_nce $info_nce \
#     --freeze_visual_tower $freeze_visual_tower \
#     --train_file $your_data_path/$TRAIN_FILE \
#     --cache_dir $your_data_path \
#     --prompt_column input \
#     --model_choice $model_choice \
#     --response_column target \
#     --overwrite_cache \
#     --dataset_name $dataset \
#     --model_name_or_path $model_name_or_path \
#     --output_dir $your_checkpopint_path/$dataset/$prefix \
#     --overwrite_output_dir \
#     --max_source_length $MAX_SOURCE_LENGTH \
#     --max_target_length 256 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --max_steps ${MAX_STEPS} \
#     --logging_steps 100 \
#     --save_steps ${MAX_STEPS} \
#     --learning_rate $LR \
#     --lora_rank ${lora_rank} \
#     --trainable ${lora_trainable} \
#     --modules_to_save ${modules_to_save} \
#     --lora_dropout ${lora_dropout} \
#     --pool_type avg \
#     --dropout_ratio 0.4 \
    # --bf16
    

# # Testing Command , seperate image and text before pooling

#  lora-mm_0325_freeze_vision_avg/checkpoint-4000 
deepspeed --master_port $MASTER_PORT main_llm.py \
    --model_choice $model_choice \
    --multimodal_embed $MULTIMODAL_EMB \
    --dataset_name $dataset \
    --enable_moelora $enable_moelora \
    --enable_task_gate $enable_task_gate \
    --concat_features $concat_features \
    --num_moe $NUM_MOE \
    --item_emb_path "./data/$dataset/handled/itm_emb_sasrec_seq.pkl" \
    --do_predict \
    --test_file $your_data_path/$TRAIN_FILE \
    --cache_dir $your_data_path \
    --overwrite_cache \
    --prompt_column input \
    --response_column target \
    --model_name_or_path $model_name_or_path \
    --peft_path $your_checkpopint_path/$dataset/$prefix/checkpoint-$MAX_STEPS \
    --output_dir results/$dataset/llm-emb \
    --output_file ${output_file}.json \
    --overwrite_output_dir \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length 196 \
    --per_device_eval_batch_size 32 \
    --predict_with_generate \
    --pool_type avg


# ## Our Method LLMEmb
gpu_id=0
seed_list=(42)
llm_emb_file="${prefix}_late_concat_pca"
# moelora_moe4_mean_0411_unfreeze_visual_late_concat_pca
# llm_emb_file="moelora_task_id-mm_0327_avg_concat_pca_late_concat"
# lvlm_image_emb_file="image_emb_lvlm"
# lvlm_text_emb_file="text_emb_lvlm"
tau=2
alpha_list=(0.01 0.05 0.1 0.15 0.2)
ts_user=9
ts_item=4


model_name="llmemb_sasrec"
for alpha in ${alpha_list[@]}
do

    python main.py --dataset ${dataset} \
                --model_name ${model_name} \
                --hidden_size 128 \
                --train_batch_size 512 \
                --max_len 200 \
                --gpu_id ${gpu_id} \
                --num_workers 8 \
                --num_train_epochs 200 \
                --seed 42 \
                --check_path "llmemb" \
                --patience 20 \
                --ts_user ${ts_user} \
                --ts_item ${ts_item} \
                --freeze_emb \
                --llm_emb_file ${llm_emb_file} \
                --alpha ${alpha} \
                --tau ${tau} \
                --log

done

model_name="llmemb_bert4rec"
for alpha in ${alpha_list[@]}
do

    python main.py --dataset ${dataset} \
                --model_name ${model_name} \
                --hidden_size 128 \
                --train_batch_size 512 \
                --max_len 200 \
                --gpu_id ${gpu_id} \
                --num_workers 8 \
                --num_train_epochs 200 \
                --seed 42 \
                --check_path "llmemb" \
                --patience 20 \
                --ts_user ${ts_user} \
                --ts_item ${ts_item} \
                --freeze_emb \
                --llm_emb_file ${llm_emb_file} \
                --alpha ${alpha} \
                --tau ${tau} \
                --log

done

