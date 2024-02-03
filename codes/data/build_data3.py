import argparse
import json
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "--name",
    nargs="?",
    default="Movielens",
    help="Choose a dataset folder from {Movielens, Tiktok}.",
)

np.random.seed(123)

args = parser.parse_args()
name = args.name
folder = name
print(name)
core = 5

if (name == "Movielens"):
    '''1. 데이터셋 분할'''

    if not os.path.exists(folder + "/%d-core" % core):
        os.makedirs(folder + "/%d-core" % core)

    df = pd.read_csv(folder + "/ml-100k/u.data", sep='\t', names=["userId", "movieId", "rating", "timestamp"])
    ui = defaultdict(list)
    for _, row in df.iterrows():
        user, item = int(row["userId"]), int(row["movieId"])
        ui[user].append(item)

    train_json, val_json, test_json = {}, {}, {}
    for user, items in ui.items():
        if len(items) < 10:
            testval = np.random.choice(len(items), 2, replace=False)
        else:
            testval = np.random.choice(len(items), int(len(items) * 0.2), replace=False)

        test = testval[: len(testval) // 2]
        val = testval[len(testval) // 2:]
        train = [i for i in range(len(items)) if i not in testval]

        train_json[user] = [items[idx] for idx in train]
        val_json[user] = [items[idx] for idx in val.tolist()]
        test_json[user] = [items[idx] for idx in test.tolist()]

    with open(folder + "/%d-core/train.json" % core, "w") as f:
        json.dump(train_json, f)
    with open(folder + "/%d-core/val.json" % core, "w") as f:
        json.dump(val_json, f)
    with open(folder + "/%d-core/test.json" % core, "w") as f:
        json.dump(test_json, f)
    '''
    # 경로 설정
    if not os.path.exists(folder + "/%d-core" % core):
        os.makedirs(folder + "/%d-core" % core)

    df = pd.read_csv(folder+ "/ratings.csv", index_col=None, usecols=None)
    ui = defaultdict(list)
    for _, row in df.iterrows():
        user, item = int(row["userId"]), int(row["movieId"])
        ui[user].append(item)

    train_json, val_json, test_json = {}, {}, {}
    for user, items in ui.items():
        if len(items) < 10:
            testval = np.random.choice(len(items), 2, replace=False)
        else:
            testval = np.random.choice(len(items), int(len(items) * 0.2), replace=False)

        test = testval[: len(testval) // 2]
        val = testval[len(testval) // 2:]
        train = [i for i in range(len(items)) if i not in testval]

        train_json[user] = [items[idx] for idx in train]
        val_json[user] = [items[idx] for idx in val.tolist()]
        test_json[user] = [items[idx] for idx in test.tolist()]

    with open(folder + "/%d-core/train.json" % core, "w") as f:
        json.dump(train_json, f)
    with open(folder + "/%d-core/val.json" % core, "w") as f:
        json.dump(val_json, f)
    with open(folder + "/%d-core/test.json" % core, "w") as f:
        json.dump(test_json, f)
    '''
    
    '''2. feature 임베딩 저장'''


elif (name == "Tiktok"):
    '''1. 데이터셋 분할'''
    # 경로 설정
    if not os.path.exists(folder + "/%d-core" % core):
        os.makedirs(folder + "/%d-core" % core)
        print("Complete making folder")

    with open(f'{folder}/train.json', 'r') as f:
        train_data = json.load(f)
    with open(f'{folder}/test.json', 'r') as f:
        test_data = json.load(f)
    with open(f'{folder}/val.json', 'r') as f:
        val_data = json.load(f)

    # Combine the data
    combined_ui = defaultdict(list)
    for dataset in [train_data, test_data, val_data]:
        for user, items in dataset.items():
            combined_ui[user].extend(items)
    combined_ui2 = {user: items for user, items in combined_ui.items() if len(items) >= core}

    # 분할 및 저장
    train_json, val_json, test_json = {}, {}, {}
    for user, items in combined_ui.items():
        if len(items) < 2:
            # 상호작용이 하나만 있는 경우, 무작위로 데이터셋 선택
            dataset_choice = np.random.choice(['train', 'val', 'test'])
            if dataset_choice == 'train':
                train_json[user] = items
                val_json[user] = []
                test_json[user] = []
            elif dataset_choice == 'val':
                val_json[user] = items  # 검증 셋에 할당
                train_json[user] = []
                test_json[user] = []
            else:
                test_json[user] = items
                train_json[user] = []
                val_json[user] = []
            continue
        elif len(items) < 10:
            testval = np.random.choice(len(items), 2, replace=False)
        else:
            testval = np.random.choice(len(items), int(len(items) * 0.2), replace=False)

        test = testval[: len(testval) // 2]
        val = testval[len(testval) // 2:]
        train = [i for i in range(len(items)) if i not in testval]

        train_json[user] = [items[idx] for idx in train]
        val_json[user] = [items[idx] for idx in val.tolist()]
        test_json[user] = [items[idx] for idx in test.tolist()]

    # Save the redistributed data
    with open(f"{folder}/{core}-core/train.json", "w") as f:
        json.dump(train_json, f)
    with open(f"{folder}/{core}-core/val.json", "w") as f:
        json.dump(val_json, f)
    with open(f"{folder}/{core}-core/test.json", "w") as f:
        json.dump(test_json, f)

    '''2. feature 임베딩 저장'''
    # 이미 저장되어 있음



