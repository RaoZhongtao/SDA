import os
import pickle
import jsonlines
import pandas as pd
import numpy as np
import json
import copy
from tqdm import tqdm

data = json.load(open("./handled/item2attributes.json", "r"))
meta_files = ['./raw/yelp_academic_dataset_business.json', './raw/yelp_academic_dataset_checkin.json', './raw/yelp_academic_dataset_review.json']
for meta_file in meta_files:
    lines = open(meta_file).readlines()
    for line in tqdm(lines):
        info = json.loads(line)
        if info['business_id'] == "9iLiMm3Z9nepRDu1AhgEoQ":
            print(f"found {info['business_id']} in {meta_file}")
            break