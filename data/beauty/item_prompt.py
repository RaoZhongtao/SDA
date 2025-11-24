import os
import pickle
import jsonlines
import pandas as pd
import numpy as np
import json
import copy
from tqdm import tqdm

data = json.load(open("./handled/item2attributes.json", "r"))
instruction = "The beauty item has the following attributes: \n "

item_data = {}
limited = 1000
count = 0
for item_dict in tqdm(data.values()):
    count += 1
    if count > limited:
        continue
    item_prompt = copy.deepcopy(instruction)
    item_id = None
    for key, value in item_dict.items():
        if key in ["related", "imUrl", "salesRank"]:   # drop longitude and latitude
            continue
        elif key in ["asin"]:  # get the item id
            item_id = value
        elif key in ["categories"]:    # list type attributes
            attri_str = ""
            for meta_str in value[0]:
                attri_str += (meta_str + ", ")
            if len(value) == 0:
                attri_str = "none, "
            attri_str = attri_str.replace("\n", " ").replace(";", ".")
            if len(attri_str) > 100:
                attri_str = attri_str[:100]
            attri_prompt = key + " is " + attri_str[:-2] + "; "    # [:-2] is to remove the last ", "
            item_prompt += attri_prompt
        else:   # str type attributes 
            if len(str(value)) > 100:
                value = value[:100]
            attri_prompt = key + " is " + str(value).replace("\n", " ").replace(";", ".") + "; "
            item_prompt += attri_prompt
    if item_id:
        item_data[item_id] = item_prompt[:-2]
    else:
        raise ValueError("No item id")
    
    
# convert to jsonline
def save_data(data_path, data):
    '''write all_data list to a new jsonl'''
    with jsonlines.open("./handled/"+ data_path, "w") as w:
        for meta_data in data:
            w.write(meta_data)

id_map = json.load(open("./handled/id_map.json", "r"))["item2id"]
json_data = []
for key, value in item_data.items():
    json_data.append({"input": value, "target": "", "item": key, "item_id": int(id_map[key])})
# 按照 item_id 进行递增排序
json_data.sort(key=lambda x: x["item_id"])
save_data("item_str_0316.jsonline", json_data)