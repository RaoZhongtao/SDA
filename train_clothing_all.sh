cd data/clothing
python item_prompt.py
cd ~/MultimodalEmb
nohup bash experiments/clothing/scft/avg.bash > logs/train_beauty_llama.log 2>&1
cd results
python convert.py
cd ~/MultimodalEmb
nohup bash experiments/clothing/rat/llmemb.bash > logs/train_clothing_llmemb.log 2>&1