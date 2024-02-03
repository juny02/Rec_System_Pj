import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



def data_split_and_save_features(path, core, has_v=True, has_a=True, has_t=True):

    train_edge = np.load(path+'/train.npy', allow_pickle=True)
    user_item_dict = np.load(path+'/user_item_dict.npy', allow_pickle=True).item()

    '''1. 데이터 셋 분할'''
    # 경로 설정
    if not os.path.exists(folder + "/%d-core" % core):
        os.makedirs(folder + "/%d-core" % core)


    train_json = {}
    val_json = {}
    test_json = {}

    for user, item_list in user_item_dict.items():
        # 각 사용자에 대한 아이템 리스트 분할
        if len(item_list) < 10:
            test_val_indices = np.random.choice(len(item_list), 2, replace=False)
        else:
            test_val_indices = np.random.choice(len(item_list), int(len(item_list) * 0.2), replace=False)

        test_indices = test_val_indices[: len(test_val_indices) // 2]
        val_indices = test_val_indices[len(test_val_indices) // 2:]

        for idx, item in enumerate(item_list):
            if idx in test_indices:
                test_json[user].append(item)
            elif idx in val_indices:
                val_json[user].append(item)
            else:
                train_json[user].append(item)

    # JSON 파일로 저장
    with open(path + "/5-core/train.json", "w") as f:
        json.dump(train_json, f)
    with open(path + "/5-core/val.json", "w") as f:
        json.dump(val_json, f)
    with open(path + "/5-core/test.json", "w") as f:
        json.dump(test_json, f)



    '''2. 3가지 feature 로드'''
    v_features, a_features, t_features = [], [], []

    if path == 'Movielens':
        num_user = 55485
        num_item = 5986
        if has_v:
            v_data = np.load(path+'/FeatureVideo_normal.npy', allow_pickle=True)
            v_features = [v_data[i] for i in range(num_item)]
        if has_a:
            a_data = np.load(path+'/FeatureAudio_avg_normal.npy', allow_pickle=True)
            a_features = [a_data[i] for i in range(num_item)]
        if has_t:
            t_data = np.load(path+'/FeatureText_stl_normal.npy', allow_pickle=True)
            t_features = [t_data[i] for i in range(num_item)]
    elif path == 'Tiktok':
        num_user = 36656
        num_item = 76085
        if has_v:
            v_data = torch.load(path+'/feat_v.pt')
            v_features = [v_data[i].numpy() for i in range(num_item)]
        if has_a:
            a_data = torch.load(path+'/feat_a.pt')
            a_features = [a_data[i].numpy() for i in range(num_item)]
        if has_t:
            t_data = torch.load(path+'/feat_t.pt')
            t_features = [t_data[i].numpy() for i in range(num_item)]

    # 각 특성을 numpy 파일로 저장
    if has_v and v_features:
        np.save(path + '/image_feat.npy', np.array(v_features, dtype=np.float32))
    if has_a and a_features:
        np.save(path + '/acoustic_feat.npy', np.array(a_features, dtype=np.float32))
    if has_t and t_features:
        np.save(path + '/text_feat.npy', np.array(t_features, dtype=np.float32))

    # return num_user, num_item, train_edge, user_item_dict


parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "--name",
    nargs="?",
    default="Movielens",
    help="Choose a dataset folder from {Movielens, Tiktok}.",
)

np.random.seed(123)

args = parser.parse_args()
folder = args.name #+ "/"
name = args.name
core = 5

if folder in ["Movielens", "Tiktok"]:
    data_split_and_save_features(folder, core)