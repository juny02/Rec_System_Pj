import heapq
import multiprocessing
import pickle
from time import time

import numpy as np
import torch
import utility.metrics as metrics
from tqdm import tqdm
from utility.load_data import Data
from utility.parser import parse_args

'''
모델 평가 위한 코드
-> 데이터셋 로드하고 모델 평가하는 함수 만듦
( 모델 평가 -> 모델에서 얻어진 최종 임베딩 기반으로 하게됨 )

1. 초기 설정, 데이터 준비
- 병렬처리 설정
- 명령어 파싱
- Data객체 -> dataset 로드 (-> 이 data generator은 main에서 쓰이게 됨)
- 평가시 필요 정보 추출

2. 평가 위해 필요 함수 정의
 A. ranklist_by_heapq, ranklist_by_sorted : rankList, auc얻음
 B. get_auc : auc계산
 C. get_performance : Precision, Recall, NDCG, Hit Ratio 지표 계산
 D. test_one_user : 한 유저에 대한 평가 수행
 
3. 최종 평가 함수
 = test_torch : 테스트할 유저 batch에 대해 평가 수행 (main)
'''

# 병렬처리 위함
cores = multiprocessing.cpu_count() // 5

# commandline argument
args = parse_args()
Ks = eval(args.Ks)

# Data생성 객체 (경로 지정해서) (load_data에서 만든 Data객체)
data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
# 유저 갯수, 아이템 갯수, 훈련 샘플 수, test 샘플 수 등 가져옴!
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test

# target aware이면 batchsize 16, 아니면 명령어값 사용
if args.target_aware:
    BATCH_SIZE = 16
else:
    BATCH_SIZE = args.batch_size

'''
heapq를 통해
1. 상위 k개 아이템 찾음
2. 그 아이템들이 실제로 선호되었는지 아닌지 1,0로 순서대로 저장
'''
def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    # 각 아이템 i들에 대한 rating 값 저장 (딕셔너리)
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    # 상위 k개 선택
    K_max = max(Ks)
    # 딕셔너리 item_score에서 가장 큰 값 가진 상위 k개 아이템 선택 (key저 값에 대해 정렬), 아이템 배열
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    # 모델에서 얻은 k개의 추천 아이템중 실제 선호하는 아이템이면이면 1, 아니면 0 저장 (배열. 순서대로)
    r = []
    for i in K_max_item_score:
        if i in user_pos_test: #선호한 아이템이면 - 배열에 1
            r.append(1)
        else:
            r.append(0)
    auc = 0.0
    return r, auc

# auc 지표 얻음
def get_auc(item_score, user_pos_test):
    # 내림차순으로 얻음
    item_score = sorted(item_score.items(), key=lambda kv: kv[1]) #값에 대해 정렬 (rating값)
    item_score.reverse()

    # 아이템 순서
    item_sort = [x[0] for x in item_score]

    # 아이템 예측 점수
    posterior = [x[1] for x in item_score]

    # 아이템 순서대로 실제로 선호된건지 아닌지 1,0 저장
    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)

    # r을 ground truth로, prediction을 posterior(선호도값) 으로 넘겨서 auc값 구함
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

'''
( 위의 rankList_by_heapq와 거의 다를게 없음.. 그치만 얘는 auc를 얻음 )
1. 상위 k개 아이템 찾음
2. 그 아이템들이 실제로 선호되었는지 아닌지 1,0로 순서대로 저장
'''
def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    # 큰거대로 정렬
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    # rankList
    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

'''
precision, recall, ndcg, hit_ratio 메틱에 대해 평가
'''
def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    # K 값마다 각 메틱 계산해서 넣음
    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    # 그걸 딕셔너리로 리턴 (string:np.array)
    return {
        "recall": np.array(recall),
        "precision": np.array(precision),
        "ndcg": np.array(ndcg),
        "hit_ratio": np.array(hit_ratio),
        "auc": auc,
    }

'''
한명의 user에 대한 테스트 진행
'''
def test_one_user(x):
    # user u's ratings for user u

    # validation set인지 (val or test set)
    is_val = x[-1]
    # 유저가 부여한 rating값
    rating = x[0]
    # uid
    u = x[1]
    # user u's items in the training set
    # training set에서 유저가 interaction한 item들
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []

    # validation set이냐 test set이느냐에 따라 각 set에서 상호작용한 item 가져옴 (정답이 됨)
    if is_val:
        user_pos_test = data_generator.val_set[u]
    else:
        user_pos_test = data_generator.test_set[u]

    # 모든 아이템 집합
    all_items = set(range(ITEM_NUM))

    # training set에 속한 아이템 제외 (test item들에 대해서만 평가해야되므로)
    test_items = list(all_items - set(training_items))

    # 명령어에 따라 둘중 어떤거 사용할지 결정됨
    # ???
    # rankList와 auc구해서 그걸로 최종 성능 구함
    if args.test_flag == "part":
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    # 성능
    return get_performance(user_pos_test, r, auc, Ks)


'''
모델 성능 테스트
매개변수 차례로
- 학습한 유저 임베딩 행렬
- 아이템 임베딩 행렬
- 테스트할 유저 배열
- validation set인지
- 인접행렬
- beta 하이퍼 파라미터
- target aware 여부
'''
def test_torch(
    ua_embeddings, ia_embeddings, users_to_test, is_val, adj, beta, target_aware
):
    # 결과 초기화
    # k의 갯수만큼 있음 [5, 10, 15] 이런식으로 있겠지요
    result = {
        "precision": np.zeros(len(Ks)),
        "recall": np.zeros(len(Ks)),
        "ndcg": np.zeros(len(Ks)),
        "hit_ratio": np.zeros(len(Ks)),
        "auc": 0.0,
    }

    # 멀티 프로세싱 위한 pool 초기화
    pool = multiprocessing.Pool(cores) # 코어 수

    # 유저, 아이템 batch size 설정
    u_batch_size = BATCH_SIZE * 2
    i_batch_size = BATCH_SIZE

    # test할 유저 목록, 갯수, batch 갯수
    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1
    count = 0

    # 아이템 임베딩 유사도 계산 (행렬곱)
    item_item = torch.mm(ia_embeddings, ia_embeddings.T)

    # 각 user batch에 대해 루프 돌림, tqdm으로 작업상황 보임
    for u_batch_id in tqdm(range(n_user_batchs), position=1, leave=False):
        # 해당 batch 부분 테스트 유저 배열에서 가져옴
        start = u_batch_id * u_batch_size # 처리 시작 인덱스
        end = (u_batch_id + 1) * u_batch_size #처리 끝 인덱스
        user_batch = test_users[start:end] # 그부분 딱 가져옴

        # target-aware을 사용하면
        if target_aware:
            # item의 batch 수
            n_item_batchs = ITEM_NUM // i_batch_size + 1
            # item에 대한 rating 저장할 행렬 (유저batch size x 전체 아이템)
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

            i_count = 0
            # item 배치에 대해 반복
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size # 아이템에서 해당 배치의 시작인덱스
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM) # 끝 인덱스

                # 현재 처리중인 item 인덱스 범위
                item_batch = range(i_start, i_end)
                # 현재 처리중인 유저 임베딩 얻음
                u_g_embeddings = ua_embeddings[user_batch]  # (batch_size, dim)
                # 현재 처리중인 item 임베딩 얻음
                i_g_embeddings = ia_embeddings[item_batch]  # (batch_size, dim)

                # target-aware
                # 1. target-orieted 유저임베딩 구함
                # 현재 처리 중인 아이템들의 행 추출 (각 아이템-다른 아이템들간의 유사도)
                item_query = item_item[item_batch, :]  # (item_batch_size, n_items)
                # softmax로 각 아이템들에 대한 중요도 계산
                item_target_user_alpha = torch.softmax(
                    torch.multiply(
                        item_query.unsqueeze(1), adj[user_batch, :].unsqueeze(0)
                    ).masked_fill( # 상호작용 없는부분 처리
                        adj[user_batch, :].repeat(len(item_batch), 1, 1) == 0, -1e9
                    ),
                    dim=2,
                )  # (item_batch_size, user_batch_size, n_items)
                # softmax로 구해진 가중치와 아이템 임베딩 곱함 -> 각 아이템들에 대한 최종 target-oriented 유저임베딩
                item_target_user = torch.matmul(
                    item_target_user_alpha, ia_embeddings
                )  # (item_batch_size, user_batch_size, dim)

                # target-aware
                # 2. 최종 matching score - batch 아이템들에 대한
                i_rate_batch = (1 - beta) * torch.matmul(
                    u_g_embeddings, torch.transpose(i_g_embeddings, 0, 1)
                ) + beta * torch.sum(
                    torch.mul(
                        item_target_user.permute(1, 0, 2).contiguous(), i_g_embeddings
                    ),
                    dim=2,
                )

                # rate_batch에 저장
                rate_batch[:, i_start:i_end] = i_rate_batch.detach().cpu().numpy()

                # 완료한 아이템들 추가
                i_count += i_rate_batch.shape[1]

                # 필요x 변수 삭제
                del (
                    item_query,
                    item_target_user_alpha,
                    item_target_user,
                    i_g_embeddings,
                    u_g_embeddings,
                )
                # GPU 캐시 비움
                torch.cuda.empty_cache()

            # 모든 아이템에 대해 되었는지 확인
            assert i_count == ITEM_NUM

        # target-aware가 아니면 -> 유저 general 임베딩만 사용하면 됨
        else:
            item_batch = range(ITEM_NUM)
            u_g_embeddings = ua_embeddings[user_batch]
            i_g_embeddings = ia_embeddings[item_batch]

            # 단순히 유저임베딩x아이템 임베딩 해주면 됨 (해당 batch에 해당하는)
            rate_batch = torch.matmul(
                u_g_embeddings, torch.transpose(i_g_embeddings, 0, 1)
            )
            rate_batch = rate_batch.detach().cpu().numpy()

        # (예측값, 유저 id, valid) 이렇게 튜플 만듦
        user_batch_rating_uid = zip(rate_batch, user_batch, [is_val] * len(user_batch))

        # 각 정보 담고있는 튜플을 test_one_user함수에 전달 -> 각 유저에 대해 test -> 저 변수는 성능 지표 값 갖게됨
        # 병렬
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        # 처리한 유저수 추가
        count += len(batch_result)

        # 결과를 평균 성능으로 집계
        # 각 유저에 대한 결과를 값/전체유저 한거를 더해서 평균 낸거와 같게 됨 !
        for re in batch_result:
            result["precision"] += re["precision"] / n_test_users
            result["recall"] += re["recall"] / n_test_users
            result["ndcg"] += re["ndcg"] / n_test_users
            result["hit_ratio"] += re["hit_ratio"] / n_test_users
            result["auc"] += re["auc"] / n_test_users

    # 모든 유저에 대해 했는지 확인
    assert count == n_test_users
    pool.close()
    return result


