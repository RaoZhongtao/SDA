source /etc/profile.d/modules.sh
conda activate MultimodalEmb
cd ~/MultimodalEmb
nohup bash experiments/toys/scft/moelora.bash > logs/train_toys_moe_4_0404.log 2>&1 &
