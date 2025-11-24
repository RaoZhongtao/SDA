import os
import pickle
import jsonlines
import pandas as pd
import numpy as np
import json
import copy
from tqdm import tqdm

data = json.load(open("./handled/item2attributes.json", "r"))

example_dict = {}
for item_dict in tqdm(data.values()):
    example_dict.update(item_dict)
    
cate_dict = {}
for item_dict in tqdm(data.values()):
    if len(item_dict["categories"]) >= 1:
        cate_dict[item_dict["business_id"]] = item_dict["categories"][-1]
    else:
        cate_dict[item_dict["business_id"]] = "NA"
        
instruction = "The point of interest has the following attributes: \n "

item_data = {}
for item_dict in tqdm(data.values()):
    item_prompt = copy.deepcopy(instruction)
    item_id = None
    for key, value in item_dict.items():
        if key in ["longitude", "latitude",]:   # drop longitude and latitude
            continue
        elif key in ["business_id"]:  # get the item id
            item_id = value
        elif key in ["categories", "neighborhoods"]:    # list type attributes
            attri_str = ""
            for meta_str in value:
                attri_str += (meta_str + ", ")
            if len(value) == 0:
                attri_str = "none, "
            attri_prompt = key + " is " + attri_str[:-2] + "; "    # [:-2] is to remove the last ", "
            item_prompt += attri_prompt
        else:   # str type attributes 
            attri_prompt = key + " is " + str(value).replace("\n", ", ") + "; "
            item_prompt += attri_prompt
    if item_id:
        item_data[item_id] = item_prompt[:-2]
    else:
        raise ValueError("No item id")
    
json.dump(item_data, open("./handled/item_str.json", "w"))

# convert to jsonline
def save_data(data_path, data):
    '''write all_data list to a new jsonl'''
    with jsonlines.open("./handled/"+ data_path, "w") as w:
        for meta_data in data:
            w.write(meta_data)

id_map = json.load(open("./handled/id_map.json", "r"))["item2id"]
json_data = []
for key, value in item_data.items():
    json_data.append({"input": value, "target": "", "item": key, "item_id": id_map[key]})

save_data("item_str.jsonline", json_data)
