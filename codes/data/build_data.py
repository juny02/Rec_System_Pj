import argparse
import array
import gzip
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# load data 파일에도 있는데 그거랑 거의 동일, 여기서는 d-core에서 core 설정 가능
def dataset_merge_and_split(path, core):
    # 존재하지 않으면 폴더 생성
    if not os.path.exists(folder + "%d-core" % core):
        os.makedirs(folder + "%d-core" % core)

    # 데이터 불러와서 ui에 다 저장
    df = pd.read_csv(path + "/train.csv", index_col=None, usecols=None)
    # Construct matrix
    ui = defaultdict(list)
    for _, row in df.iterrows():
        user, item = int(row["userID"]), int(row["itemID"])
        ui[user].append(item)

    df = pd.read_csv(path + "/test.csv", index_col=None, usecols=None)
    for _, row in df.iterrows():
        user, item = int(row["userID"]), int(row["itemID"])
        ui[user].append(item)

    # 분할
    train_json = {}
    val_json = {}
    test_json = {}
    for u, items in ui.items():
        if len(items) < 10:
            testval = np.random.choice(len(items), 2, replace=False)
        else:
            testval = np.random.choice(len(items), int(len(items) * 0.2), replace=False)

        test = testval[: len(testval) // 2]
        val = testval[len(testval) // 2 :]
        train = [i for i in list(range(len(items))) if i not in testval]
        train_json[u] = [items[idx] for idx in train]
        val_json[u] = [items[idx] for idx in val.tolist()]
        test_json[u] = [items[idx] for idx in test.tolist()]

    # 여긴 d 설정할수잇엇는데 왜 또 5로 되어잇지
    with open(path + "/5-core/train.json", "w") as f:
        json.dump(train_json, f)
    with open(path + "/5-core/val.json", "w") as f:
        json.dump(val_json, f)
    with open(path + "/5-core/test.json", "w") as f:
        json.dump(test_json, f)


def load_textual_image_features(data_path):
    import json
    import os

    from gensim.models.doc2vec import Doc2Vec

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


'''
[ 데이터처리 - 데이터를 불러와서 적합한 형식으로 변환 ]
모델 돌리기 전에, 데이터 처리 먼저 해주어야함
-> python build_data.py --name={Dataset} 이런식으로 원하는 데이터셋에 대해 우선 data build해줌

- Clotihng dataset과 다른 데이터셋은 처리 경로가 다름
- Men, WomenClothing
    - dataset_merge_and_split(folder, core) -> 데이터셋 셋으로 나눔
    - load_textual_image_features(folder) -> 이미지, 텍스트 정보 추출
- 아닌 경우 -> 추가 처리가 더 필요한 원시 데이터 인듯 
    ( reviews data, meta data, image features 을 토대로 추출해내야함 )
    1. 버트모델 -> 텍스트 데이터로 특성 추출
    2. 이미지 특성 추출
    3. 상호작용 데이터 분석
    
어쨌든 결론적으로는 불려진 데이터셋에 대해 
- path + "/5-core/train.json, val, test로 interaction data 세개로 나눔
- data_path + "/text_feat.npy", data_path + "/image_feat.npy" 파일 에 text, image 특성 저장
'''

# argumentParser 객체 생성
parser = argparse.ArgumentParser(description="")
# argument 설정 (dataset 이름)
parser.add_argument(
    "--name",
    nargs="?",
    default="MenClothing",
    help="Choose a dataset folder from {MenClothing, WomenClothing, Beauty, Toys_and_Games}.",
)
# 시드 고정
np.random.seed(123)

args = parser.parse_args()
# 데이터셋 경로, 이름 설정
folder = args.name + "/"
name = args.name
core = 5

# 해당 데이터셋에 대한거면 데이터 불러오고 분할, 텍스트 이미지 특성 추출함
if folder in ["MenClothing/", "WomenClothing/"]:
    dataset_merge_and_split(folder, core)
    load_textual_image_features(folder)
else:
    # 버트 기반 언어 모델 로드
    bert_path = "sentence-transformers/stsb-roberta-large"
    bert_model = SentenceTransformer(bert_path)

    # 여기에 저장해주게 됨
    if not os.path.exists(folder + "%d-core" % core):
        os.makedirs(folder + "%d-core" % core)

    # 데이터들 JSON형식으로 파싱
    def parse(path):
        g = gzip.open(path, "r")
        for line in g:
            yield json.dumps(eval(line))

    # 메타데이터 파싱
    print("----------parse metadata----------")
    if not os.path.exists(folder + "meta-data/meta.json"):
        # 여기 저장
        with open(folder + "meta-data/meta.json", "w") as f:
            for line in parse(folder + "meta-data/" + "meta_%s.json.gz" % (name)):
                f.write(line + "\n")

    # 리뷰데이터 파싱
    print("----------parse data----------")
    if not os.path.exists(folder + "meta-data/%d-core.json" % core):
        with open(folder + "meta-data/%d-core.json" % core, "w") as f:
            for line in parse(
                folder + "meta-data/" + "reviews_%s_%d.json.gz" % (name, core)
            ):
                f.write(line + "\n")

    # 데이터 로드 (위에서 파싱한 리뷰데이터)
    print("----------load data----------")
    jsons = []
    for line in open(folder + "meta-data/%d-core.json" % core).readlines():
        jsons.append(json.loads(line))

    # 아이템, 유저 정보 추출, 상호작용 맵핑
    print("----------Build dict----------")
    # 유저, 아이템 집합 생성
    items = set()
    users = set()
    for j in jsons:
        items.add(j["asin"])
        users.add(j["reviewerID"])
    print("n_items:", len(items), "n_users:", len(users))

    # 유저, 아이템 id 할당
    item2id = {}
    with open(folder + "%d-core/item_list.txt" % core, "w") as f:
        for i, item in enumerate(items):
            item2id[item] = i
            f.writelines(item + "\t" + str(i) + "\n")

    user2id = {}
    with open(folder + "%d-core/user_list.txt" % core, "w") as f:
        for i, user in enumerate(users):
            user2id[user] = i
            f.writelines(user + "\t" + str(i) + "\n")

    # 상호작용 딕셔너리 생성
    ui = defaultdict(list)
    review2id = {}
    review_text = {}
    ratings = {}
    # 리뷰 텍스트 -> rating값 맵핑?
    with open(folder + "%d-core/review_list.txt" % core, "w") as f:
        for j in jsons:
            u_id = user2id[j["reviewerID"]]
            i_id = item2id[j["asin"]]
            ui[u_id].append(i_id)  # ui[u_id].append(i_id)
            review_text[len(review2id)] = j["reviewText"].replace("\n", " ")
            ratings[len(review2id)] = int(j["overall"])
            f.writelines(str((u_id, i_id)) + "\t" + str(len(review2id)) + "\n")
            review2id[u_id, i_id] = len(review2id)
    with open(folder + "%d-core/user-item-dict.json" % core, "w") as f:
        f.write(json.dumps(ui))
    with open(folder + "%d-core/rating-dict.json" % core, "w") as f:
        f.write(json.dumps(ratings))

    review_texts = []
    with open(folder + "%d-core/review_text.txt" % core, "w") as f:
        for i, j in review2id:
            f.write(review_text[review2id[i, j]] + "\n")
            review_texts.append(review_text[review2id[i, j]] + "\n")
    review_embeddings = bert_model.encode(review_texts)
    assert review_embeddings.shape[0] == len(review2id)
    np.save(folder + "review_feat.npy", review_embeddings)

    # 상호작용 데이터 3개로 나눠
    print("----------Split Data----------")
    train_json = {}
    val_json = {}
    test_json = {}
    for u, items in ui.items():
        if len(items) < 10:
            testval = np.random.choice(len(items), 2, replace=False)
        else:
            testval = np.random.choice(len(items), int(len(items) * 0.2), replace=False)

        test = testval[: len(testval) // 2]
        val = testval[len(testval) // 2 :]
        train = [i for i in list(range(len(items))) if i not in testval]
        train_json[u] = [items[idx] for idx in train]
        val_json[u] = [items[idx] for idx in val.tolist()]
        test_json[u] = [items[idx] for idx in test.tolist()]

    with open(folder + "%d-core/train.json" % core, "w") as f:
        json.dump(train_json, f)
    with open(folder + "%d-core/val.json" % core, "w") as f:
        json.dump(val_json, f)
    with open(folder + "%d-core/test.json" % core, "w") as f:
        json.dump(test_json, f)

    jsons = []
    with open(folder + "meta-data/meta.json", "r") as f:
        for line in f.readlines():
            jsons.append(json.loads(line))

    # 텍스트 특징 추출
    print("----------Text Features----------")
    raw_text = {}
    for _json in jsons:
        if _json["asin"] in item2id:
            string = " "
            if "categories" in _json:
                for cates in _json["categories"]:
                    for cate in cates:
                        string += cate + " "
            if "title" in _json:
                string += _json["title"]
            if "brand" in _json:
                string += _json["title"]
            if "description" in _json:
                string += _json["description"]
            raw_text[item2id[_json["asin"]]] = string.replace("\n", " ")
    texts = []
    with open(folder + "%d-core/raw_text.txt" % core, "w") as f:
        for i in range(len(item2id)):
            f.write(raw_text[i] + "\n")
            texts.append(raw_text[i] + "\n")
    sentence_embeddings = bert_model.encode(texts)
    assert sentence_embeddings.shape[0] == len(item2id)
    np.save(folder + "text_feat.npy", sentence_embeddings)


    # 이미지 특성 추출
    print("----------Image Features----------")

    def readImageFeatures(path):
        f = open(path, "rb")
        while True:
            asin = f.read(10).decode("UTF-8")
            if asin == "":
                break
            a = array.array("f")
            a.fromfile(f, 4096)
            yield asin, a.tolist()

    data = readImageFeatures(folder + "meta-data/" + "image_features_%s.b" % name)
    feats = {}
    avg = []
    for d in data:
        if d[0] in item2id:
            feats[int(item2id[d[0]])] = d[1]
            avg.append(d[1])
    avg = np.array(avg).mean(0).tolist()

    ret = []
    for i in range(len(item2id)):
        if i in feats:
            ret.append(feats[i])
        else:
            ret.append(avg)

    assert len(ret) == len(item2id)
    np.save(folder + "image_feat.npy", np.array(ret))

# 어쿠스틱 전처리 해야할 ,,듓