import json
import os

# args = parse_args()
import random as rd

# from utility.parser import parse_args
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from gensim.models.doc2vec import Doc2Vec

'''
[ Data Class ]
주어진 경로에서 데이터 로딩, 처리 하는 클래스
- 존재 변수들
    - path
    ...

- batch_test 에서 객체 생성 및 초기화 ( data_generator )
- main에서 행렬 정보 얻을 때와 훈련시 샘플링할때 사용

1. __init__ : 객체 초기화, 데이터 로드 -> batch_test에서
- parameter 
    - path : 경로
    - batch size
- training, validation, test 경로 설정 (train, test, val)
- training, validation, test set 으로 정보 얻기
    - 유저수, 아이템수, 존재 유저 (모두를 통해) (self.n_users, self.n_items, self.exist_users)
    - 각각의 상호작용 수 self.n_train, self.n_test, self.n_val
    - 각각의 상호작용 정보 self.train_items, self.test_set, self.val_set
- 행렬 R 초기화 -> interaction 저장
- 통계 정보 출력

2. nonzero_idx -> main에서 
interaction 행렬 R에서 0이아닌 인덱스 반환 ((r,c) 리스트로 )

3. sample : 배치 샘플링  -> main에서
데이터 샘플링함
- 주어진 배치 크기에 따라 유저 무작위 선택
- 각 유저에 대해 positive, negative 아이템 샘플링

4. print_statistics
데이터의 통계적 정보 출력
- self.n_users, self.n_items : 전체 유저, 아이템 수
- self.n_train + self.n_val + self.n_test : 각 set에서 상호작용 수
- sparsity : 상호작용수/전체(u x i)

---
위 Data class 사용 전 이 두 함수를 호출해야할듯? - build data에서 함

A. dataset_merge_and_split : 데이터 불러와서 train, val, test set으로 구분
- parameter : path
- 과정
    - 기존 train, test.csv에서 모든 유저, 아이템들 상호작용 정보 가져오고 이를 재분할 함
    - 데이터 분할은
        - test 10%, val 10%, 나머지 train으로 
        
B. load_textual_image_features : text, image 특성 로드, 저장
- parameter : path
'''

'''

1. __init__ : 객체 초기화, 데이터 로드
- parameter 
    - path : 경로
    - batch size
- training, validation, test 경로 설정
- 유저, 아이템수 카운트
- 최대 인덱스 추출?
- 행렬 R 초기화 -> interaction 저장
- 통계 정보 출력
'''
class Data(object):
    def __init__(self, path, batch_size):
        # 데이터 경로 저장 및 batch size 저장
        self.path = path + "/5-core" # 최소 5회 상호작용 데이터?인듯?
        self.batch_size = batch_size

        # 각 훈련, val, test 파일 경로 설정
        train_file = path + "/5-core/train.json"
        val_file = path + "/5-core/val.json"
        test_file = path + "/5-core/test.json"

        # get number of users and items
        # 우선 초기화
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test, self.n_val = 0, 0, 0
        self.neg_pools = {}

        self.exist_users = []

        # 데이터셋 로드
        # 각각 유저id:상호작용한 아이템 으로 저장됨
        train = json.load(open(train_file))
        test = json.load(open(test_file))
        val = json.load(open(val_file))

        # 각 데이터셋에 대해 작업 수행
        # train set - 유저와 아이템 수, 유저 목록, 상호작용 수
        for uid, items in train.items(): # 튜플화, 각각 uid, 그 유저가 상호작용한 아이템배열 저장
            # 상호작용한 item 없는 유저 건너뜀
            if len(items) == 0:
                continue
            uid = int(uid)
            # 유저 목록에 추가
            self.exist_users.append(uid)
            # 아이템, 유저 갯수는 최대 id값으로 업데이트함
            self.n_items = max(self.n_items, max(items))
            self.n_users = max(self.n_users, uid)
            # 상호작용 갯수
            self.n_train += len(items)

        # test set 처리 - 아이템수 업데이트, 테스트셋의 상호작용 수
        for uid, items in test.items():
            uid = int(uid)
            try:
                # 최고인덱스로 아이템수 갱신
                self.n_items = max(self.n_items, max(items))
                # 테스트셋의 상호작용수
                self.n_test += len(items)
            except Exception:
                continue
        # validation set 처리 - 아이템수 업데이트, 테스트셋의 상호작용 수 (동일)
        for uid, items in val.items():
            uid = int(uid)
            try:
                self.n_items = max(self.n_items, max(items))
                self.n_val += len(items)
            except Exception:
                continue

        # 0으로 시작하는 인덱스 고려인듯
        self.n_items += 1
        self.n_users += 1

        # 통계 정보 출력
        self.print_statistics()

        # 전체 유저, 아이템에 대한 행렬 (희소행렬)
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        # 딕셔너리 초기화
        self.train_items, self.test_set, self.val_set = {}, {}, {}
        # 각 데이터셋에서 상호작용한 아이템 목록 얻음
        # train set을 돌면서 상호작용한 유저-아이템에 대해 1 넣음
        for uid, train_items in train.items(): # 유저, 상호작용한 아이템목록
            if len(train_items) == 0:
                continue
            uid = int(uid)
            for _, i in enumerate(train_items):
                self.R[uid, i] = 1.0 # 행렬에 추가
            # 이 딕셔너리에도 유저 - 상호작용한 아이템 추가함
            self.train_items[uid] = train_items

        # test set 처리 - 유저별 상호작용한 아이템수 저장
        for uid, test_items in test.items():
            uid = int(uid)
            if len(test_items) == 0:
                continue
            try:
                self.test_set[uid] = test_items
            except Exception:
                continue

        # validation set 처리 - 유저별 상호작용한 아이템수 저장
        for uid, val_items in val.items():
            uid = int(uid)
            if len(val_items) == 0:
                continue
            try:
                self.val_set[uid] = val_items
            except Exception:
                continue

    '''
    2. nonzero_idx - main에서 사용
    interaction 행렬 R에서 0이아닌 인덱스 반환 ((r,c) 리스트로 )
    '''
    def nonzero_idx(self):
        r, c = self.R.nonzero()
        idx = list(zip(r, c))
        return idx

    '''
    3. sample : 배치 샘플링 - main에서 훈련시 사용
    데이터 샘플링함
    - 주어진 배치 크기에 따라 유저 무작위 선택
    - 각 유저에 대해 positive, negative 아이템 샘플링
    '''
    def sample(self):
        # 배치 크기가 전체 유저보다 작으면 무작위로 배치만큼 샘플링
        if self.batch_size <= self.n_users:
            print(self.n_users)
            users = rd.sample(self.exist_users, self.batch_size)
        else: # 배치 크기가 더 작으면 중복 허용 샘플링
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
        # users = self.exist_users[:]

        '''
        특정 유저 u에 대한 num개의 positive 아이템 샘플링
        - train_items[u]에서 무작위로 아이템 선택 (중복x)
        '''
        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            # 여기 저장해서 리턴
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                # 인덱스
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                # 아이템 id
                pos_i_id = pos_items[pos_id]

                # 중복 피하기
                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch
        '''
        특정 유저 u에 대한 num개의 negative 아이템 샘플링
        - 난수발생 -> train_item에 없고 중복아니면 추가
        '''
        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                # train_item에 없어야함 ! 그리고 중복 x
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        # 최종 -> 모든 유저에 대해 부정, 긍정 한개씩 샘플링함
        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
        return users, pos_items, neg_items
    '''
    4. print_statistics
    데이터의 통계적 정보 출력
    - self.n_users, self.n_items : 전체 유저, 아이템 수
    - self.n_train + self.n_val + self.n_test : 각 set에서 상호작용 수
    - sparsity : 상호작용수/전체(u x i)
    '''
    def print_statistics(self):
        print("n_users=%d, n_items=%d" % (self.n_users, self.n_items))
        print("n_interactions=%d" % (self.n_train + self.n_val + self.n_test))
        print(
            "n_train=%d, n_val=%d, n_test=%d, sparsity=%.5f"
            % (
                self.n_train,
                self.n_val,
                self.n_test,
                (self.n_train + self.n_val + self.n_test)
                / (self.n_users * self.n_items),
            )
        )

'''
A. dataset_merge_and_split : 데이터 불러와서 train, val, test set으로 구분
( 이걸 하고 Data class 작업 할 수 있을 듯)
- parameter : path
- 과정
    - 기존 train, test.csv에서 모든 유저, 아이템들 상호작용 정보 가져오고 이를 재분할 함
    - 데이터 분할은
        - test 10%, val 10%, 나머지 train으로 
'''

def dataset_merge_and_split(path):
    # train.csv - 훈련데이터 -> 불러옴
    df = pd.read_csv(path + "/train.csv", index_col=None, usecols=None)
    # Construct matrix
    ui = defaultdict(list)
    # 유저별로 아이템 리스트로 저장
    for _, row in df.iterrows():
        user, item = int(row["userID"]), int(row["itemID"])
        ui[user].append(item)

    # test data도 동일하게 ui에 다 넣어줌
    df = pd.read_csv(path + "/test.csv", index_col=None, usecols=None)
    for _, row in df.iterrows():
        user, item = int(row["userID"]), int(row["itemID"])
        ui[user].append(item)

    # 데이터를 재 분할 하는듯
    # 데이터 세트 초기화
    train_json = {}
    val_json = {}
    test_json = {}

    # 데이터 분할
    for u, items in ui.items(): # 유저 - 상호작용한 아이템들
        if len(items) < 10: # 상호작용 10 미만이면
            # 2개 아이템 뽑음
            testval = np.random.choice(len(items), 2, replace=False)
        else:
            # 아니면 10%를 테스트로 뽑음
            testval = np.random.choice(len(items), int(len(items) * 0.2), replace=False)

        # test와 validation은 절반씩 나눔
        test = testval[: len(testval) // 2]
        val = testval[len(testval) // 2 :]
        # train은 위 둘 (testval)에 없는 아이템 가짐
        train = [i for i in list(range(len(items))) if i not in testval]
        # 각각 잘 추가해줌
        train_json[u] = [items[idx] for idx in train]
        val_json[u] = [items[idx] for idx in val.tolist()]
        test_json[u] = [items[idx] for idx in test.tolist()]

    # 각 데이터셋 파일로 저장함
    with open(path + "/5-core/train.json", "w") as f:
        json.dump(train_json, f)
    with open(path + "/5-core/val.json", "w") as f:
        json.dump(val_json, f)
    with open(path + "/5-core/test.json", "w") as f:
        json.dump(test_json, f)

'''     
B. load_textual_image_features : text, image 특성 로드, 저장
- parameter : path
'''

def load_textual_image_features(data_path):
    # asin 딕셔너리 로드, 텍스트 벡터 추출
    asin_dict = json.load(open(os.path.join(data_path, "asin_sample.json"), "r"))

    # Prepare textual feture data.
    doc2vec_model = Doc2Vec.load(os.path.join(data_path, "doc2vecFile"))
    vis_vec = np.load(
        os.path.join(data_path, "image_feature.npy"), allow_pickle=True
    ).item()
    text_vec = {}
    for asin in asin_dict:
        text_vec[asin] = doc2vec_model.docvecs[asin]

    all_dict = {}
    num_items = 0
    filename = data_path + "/train.csv"
    df = pd.read_csv(filename, index_col=None, usecols=None)
    for _, row in df.iterrows():
        asin, i = row["asin"], int(row["itemID"])
        all_dict[i] = asin
        num_items = max(num_items, i)
    filename = data_path + "/test.csv"
    df = pd.read_csv(filename, index_col=None, usecols=None)
    for _, row in df.iterrows():
        asin, i = row["asin"], int(row["itemID"])
        all_dict[i] = asin
        num_items = max(num_items, i)

    t_features = []
    v_features = []
    for i in range(num_items + 1):
        t_features.append(text_vec[all_dict[i]])
        v_features.append(vis_vec[all_dict[i]])

    np.save(data_path + "/text_feat.npy", np.asarray(t_features, dtype=np.float32))
    np.save(data_path + "/image_feat.npy", np.asarray(v_features, dtype=np.float32))
