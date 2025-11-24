## General Model without any enhancement
gpu_id=0
dataset="tmall"
seed_list=(42)

model_name="sasrec_seq"
for seed in ${seed_list[@]}
do
    python main.py --dataset ${dataset} \
                --model_name ${model_name} \
                --hidden_size 128 \
                --train_batch_size 512 \
                --max_len 200 \
                --gpu_id ${gpu_id} \
                --num_workers 8 \
                --num_train_epochs 200 \
                --seed ${seed} \
                --check_path "" \
                --patience 20 \
                --ts_user 5 \
                --ts_item 2 \
                --log
done

for seed in ${seed_list[@]}
do
    python main.py --dataset ${dataset} \
                --model_name ${model_name} \
                --hidden_size 128 \
                --train_batch_size 512 \
                --max_len 200 \
                --gpu_id ${gpu_id} \
                --num_workers 8 \
                --num_train_epochs 200 \
                --seed ${seed} \
                --check_path "" \
                --patience 20 \
                --ts_user 5 \
                --ts_item 2 \
                --do_emb
done


# model_name="bert4rec"
# for seed in ${seed_list[@]}
# do
#     python main.py --dataset ${dataset} \
#                 --model_name ${model_name} \
#                 --hidden_size 128 \
#                 --train_batch_size 512 \
#                 --max_len 200 \
#                 --gpu_id ${gpu_id} \
#                 --num_workers 8 \
#                 --num_train_epochs 200 \
#                 --seed ${seed} \
#                 --check_path "" \
#                 --patience 20 \
#                 --ts_user 5 \
#                 --ts_item 2 \
#                 --log
# done

# for seed in ${seed_list[@]}
# do
#     python main.py --dataset ${dataset} \
#                 --model_name ${model_name} \
#                 --hidden_size 128 \
#                 --train_batch_size 512 \
#                 --max_len 200 \
#                 --gpu_id ${gpu_id} \
#                 --num_workers 8 \
#                 --num_train_epochs 200 \
#                 --seed ${seed} \
#                 --check_path "" \
#                 --patience 20 \
#                 --ts_user 5 \
#                 --ts_item 2 \
#                 --do_emb
# done