import numpy as np
from sklearn.metrics import roc_auc_score

# 실제 선호하는 전체 아이템중 추천한게 몇개인지
def recall(rank, ground_truth, N):
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))

# precision = 모델이 true라고 한 것 중 실제로 true인 것의 개수
# relevance가 binary로 나타날 때 (실젯값)  k개중에서 실제로 1인것의 갯수
def precision_at_k(r, k):
    """Score is precision @ k.

    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    # r의 상위 k개 자름
    r = np.asarray(r)[:k]
    # 평균냄 -> 추천한것중 실제로 선호된것의 갯수 쉽게구해짐
    return np.mean(r)


def average_precision(r, cut):
    """Score is average precision (area under PR curve).

    Relevance is binary (nonzero is relevant).
    Returns:
        Average precision
    """
    r = np.asarray(r)
    # 1~cut을 k로 설정하면서 각 precision값 계산하고 평균냄
    out = [precision_at_k(r, k + 1) for k in range(cut) if r[k]]
    if not out:
        return 0.0
    return np.sum(out) / float(min(cut, np.sum(r)))

# 여러 유저들에 대해 average_precision계산하고 이를 평균냄
def mean_average_precision(rs):
    """Score is mean average precision.

    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg).

    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    # r을 상위k개 자름
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0: # 첫번째항을 Log2 2로 안나눔
            # 아이템과의 관련성(바이너리값) /랭킹 의 합을 구함
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1: #첫번째항도 log 2 2로 나눔
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError("method must be 0 or 1.")
    return 0.0

# normalized DCG
def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg).

    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    # dcg를 구함 (r을 내림차순으로 정렬해서 구함) (이게 최대 DCG값이 됨)
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.0
    # 구한 실제 dcg값을 최댓값으로 나눠줌
    return dcg_at_k(r, k, method) / dcg_max


# 전체 선호하는 아이템중 추천한게 몇개인지
def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    if all_pos_num == 0:
        return 0
    else: # 그중에서 추천하는거 갯수. 모델이 예측한 순서대로 있음
        return np.sum(r) / all_pos_num

# 상위k개중 실제로 선호하는게 하나라도 있는지
def hit_at_k(r, k):
    r = np.array(r)[:k]
    # 하나라도 있으면 1.0
    if np.sum(r) > 0:
        return 1.0
    else:
        return 0.0

# precision과 recall값의 조화평균 (둘의 균형) -> 1에 가까울수록 좋음
def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.0


def auc(ground_truth, prediction): # 각각 추천하는 아이템 순서대로 실제로 선호여부와 그들에 대한 평가 점수배열
    try: # 그냥 내제된 함수로 계산 - 1에 가까울수록 좋음
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.0
    return res

# r 배열은 모델이 추천한 아이템 순서대로 실제 선호했는지 아닌지를 1,0으로 담는 배열