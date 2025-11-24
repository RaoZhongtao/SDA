dataset="beauty"
gpu_id=0
seed_list=(42)
tau=2
alpha_list=(0.01)
ts_user=9
ts_item=4
llm_emb_file="mm_0320_avg_seperate_concat_pca"
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
