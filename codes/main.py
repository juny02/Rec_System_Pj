import math
import random
import sys
from time import time

import numpy as np
import torch
import torch.optim as optim
from Models import MONET

# 저것들을 사용
from utility.batch_test import data_generator, test_torch
from utility.parser import parse_args

device = torch.device("mps")

'''
[ Trainer 클래스 ]
모델 훈련과 평가를 수행하는 클래스
- 모델과 모델에 대한 parameter 가짐
- 모델 초기화, 훈련, 평가 위한 함수 가짐

1. init - 모델 초기화
- parameter : 
    - config (유저수, 아이템수, 상호작용 nonzero 인덱스 가짐) 와 
    - args : 명령어-학습설정에 필요한 값
- 다음 값들을 받아 모델 초기화함

2. set_lr_scheduler - 학습률 스케줄러 설정
- 학습 과정에서 학습률 동적으로 조정위함
- 에폭따라 학습률 감소하도록

3. test - 모델 성능 평가
- parameter :
    - users_to_test: 테스트할 유저 목록
    - is_val : validation set 여부
- batch test에서 정의한 test_torch 로 모델 성능지표 계산함

4. train - 모델 학습
- 인접 행렬 생성
- 학습 과정에서 에폭별로 반복
    - 배치에 대해 모델 학습
    - 손실 계산 -> backpropagation
    - validation에 대한 모델 평가함
    - 조기 종료 로직있음 (에폭 계속되는데도 성능 개선 x일시)
- 최종 모델 저장
- test set에서 평가

5. main 실행 블록
- commandline argument 파싱
- 시드 설정
- 모델 train

'''

class Trainer(object):
    '''
    1. init - 모델 초기화
    - parameter :
        - config (유저수, 아이템수, 상호작용 nonzero 인덱스 가짐) 와
        - args : 명령어-학습설정에 필요한 값
    - 다음 값들을 받아 모델 초기화함
    '''
    def __init__(self, data_config, args):
        # argument settings
        # 설정 초기화
        self.n_users = data_config["n_users"]
        self.n_items = data_config["n_items"]

        self.feat_embed_dim = args.feat_embed_dim
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.n_layers = args.n_layers
        self.has_norm = args.has_norm
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.lamb = self.regs[1]
        self.alpha = args.alpha
        self.beta = args.beta
        self.dataset = args.dataset
        self.model_name = args.model_name
        self.agg = args.agg
        self.target_aware = args.target_aware
        self.cf = args.cf
        self.cf_gcn = args.cf_gcn
        self.lightgcn = args.lightgcn

        self.use_image = args.use_image
        self.use_txt = args.use_txt
        self.use_acoustic = args.use_acoustic
        self.use_interact = args.use_interact

        self.nonzero_idx = data_config["nonzero_idx"]

        self.image_feats = np.load("data/{}/image_feat.npy".format(self.dataset))
        self.text_feats = np.load("data/{}/text_feat.npy".format(self.dataset))
        self.acoustic_feats = np.load("data/{}/acoustic_feat.npy".format(self.dataset))
        #self.interaction_feats = np.load("data/{}/interaction_feat.npy".format(self.dataset))

        # 모델
        self.model = MONET(
            self.n_users,
            self.n_items,
            self.feat_embed_dim,
            self.nonzero_idx,
            self.has_norm,
            self.image_feats,
            self.text_feats,
            self.acoustic_feats,
            #self.interaction_feats,
            self.n_layers,
            self.alpha,
            self.beta,
            self.agg,
            self.cf,
            self.cf_gcn,
            self.lightgcn,
            self.use_image,
            self.use_txt,
            self.use_acoustic,
            self.use_interact,

        )

        self.model = self.model#cuda()
        #Adam optimizer 사용 - gradient 기반
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # 스케줄러 설정
        self.lr_scheduler = self.set_lr_scheduler()

    '''
    2. set_lr_scheduler - 학습률 스케줄러 설정
    - 학습 과정에서 학습률 동적으로 조정위함
    - 에폭따라 학습률 감소하도록
    '''
    def set_lr_scheduler(self):
        # 에폭이 증가할수록 학습률 감소하도록 (50 에포크마다 학습률을 0.96씩 제곱하여 감소시킴)
        fac = lambda epoch: 0.96 ** (epoch / 50)
        # 위에서 정의한 람다함수에 따라 학습률 조절해줌
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        return scheduler

    '''
    3. test - 모델 성능 평가
    - parameter :
        - users_to_test: 테스트할 유저 목록
        - is_val : validation set 여부
    - batch test에서 정의한 test_torch 로 모델 성능지표 계산함
    '''
    def test(self, users_to_test, is_val):
        # evaluation mode
        self.model.eval()
        # gradient computaion 안하도록 (파라미터 업데이트 xx)
        with torch.no_grad():
            # 모델에서 임베딩 얻음
            ua_embeddings, ia_embeddings = self.model()

        # 정의했던 이 함수로 성능 계산하게됨
        result = test_torch(
            ua_embeddings,
            ia_embeddings,
            users_to_test,
            is_val,
            self.adj,
            self.beta,
            self.target_aware,
        )
        return result

    '''
    4. train - 모델 학습
    - 인접행렬 생성
    - 학습 과정에서 에폭별로 반복
        - 배치에 대해 모델 학습
        - 손실 계산 -> backpropagation
        - validation에 대한 모델 평가함
        - 조기 종료 로직있음 (에폭 계속되는데도 성능 개선 x일시)
    - 최종 모델 저장
    - test set에서 평가
    '''
    def train(self):
        # 1. 인접 행렬 생성 (희소행렬 -> dense 행렬로 저장)
        # nonzero_idx는 0아닌 인덱스만 저장한 것 -> 그자리에 1로 해서 dense행렬로 만듦
        nonzero_idx = torch.tensor(self.nonzero_idx).long().T #.cuda().long().T
        self.adj = (
            torch.sparse.FloatTensor(
                nonzero_idx,
                torch.ones((nonzero_idx.size(1))),#cuda(),
                (self.n_users, self.n_items),
            )
            .to_dense()
            #cuda()
        )

        # 2. 초깃값 설정
        # 조기 종료를 위한값
        stopping_step = 0
        # 배치 수
        n_batch = data_generator.n_train // args.batch_size + 1
        # 최고 recall값 추적
        best_recall = 0

        # 3. 에폭 반복
        for epoch in range(args.epoch):
            # 에폭 시작 시간
            t1 = time()
            # 초기 손실값 초기화
            loss, mf_loss, emb_loss, reg_loss = 0.0, 0.0, 0.0, 0.0
            # 배치 수 (왜 또선언하지..)
            n_batch = data_generator.n_train // args.batch_size + 1
            # 각 에폭에 대해 학습 수행함
            for _ in range(n_batch):
                # 모델을 학습 모드로 설정
                self.model.train()
                # optimizer의 gradient초기화
                self.optimizer.zero_grad()

                # 모델에서 임베딩 얻음
                user_emb, item_emb = self.model()
                # 데이터를 얻음
                users, pos_items, neg_items = data_generator.sample()

                # 해당 데이터에 대해 손실 계산 (bpr loss)
                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.model.bpr_loss(
                    user_emb, item_emb, users, pos_items, neg_items, self.target_aware
                )

                # 임베딩 loss에는 decay 적용
                batch_emb_loss = self.decay * batch_emb_loss
                # 총 손실
                batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss

                # backpropagation으로 학습
                batch_loss.backward(retain_graph=True)
                self.optimizer.step()

                # 각 배치에 대한 손실을 전체 누적함 (각 에폭동안 얼마나 발생햇는지)
                loss += float(batch_loss)
                mf_loss += float(batch_mf_loss)
                emb_loss += float(batch_emb_loss)
                reg_loss += float(batch_reg_loss)

                # 유저, 아이템임베딩 삭제 ^^
                del user_emb, item_emb
                #torch.cuda.empty_cache()

            # 각 에폭이 끝난후 학습률 업데이트
            self.lr_scheduler.step()

            # 현재 에폭 손실이 NaN (not a number) 이면 문제잇으므로 exit
            if math.isnan(loss):
                print("ERROR: loss is nan.")
                sys.exit()

            # 각 에폭에서의 에폭 번호, 걸린 시간, 총, 각 손실을 출력함
            perf_str = "Pre_Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]" % (
                epoch,
                time() - t1,
                loss,
                mf_loss,
                emb_loss,
                reg_loss,
            )
            print(perf_str)

            # 지정한 주기? 마다 밑 코드를 실행하게됨 - test
            if epoch % args.verbose != 0:
                continue

            # 검증 시작 시간
            t2 = time()
            # test set에서 모든 유저 반환 - 왜 지금부르지... ? 구조때문인듯 . ...?
            users_to_test = list(data_generator.test_set.keys())
            # validationset에서 모든 유저 반환
            users_to_val = list(data_generator.val_set.keys())
            # validation set에 대해 test함
            ret = self.test(users_to_val, is_val=True)
            # 검증 끝 시간
            t3 = time()

            # verbose가 설정되어있으면 - 검증 결과 출력
            # 전체 에폭 소요 시간, 검증과정 소요 시간, loss와 현재 에폭의 성능 지표들 출력
            if args.verbose > 0:
                perf_str = (
                    "Pre_Epoch %d [%.1fs + %.1fs]:  val==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], "
                    "precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]"
                    % (
                        epoch,
                        t2 - t1,
                        t3 - t2,
                        loss,
                        mf_loss,
                        emb_loss,
                        reg_loss,
                        ret["recall"][0], # 낮은순위
                        ret["recall"][-1], # 높은 순위
                        ret["precision"][0],
                        ret["precision"][-1],
                        ret["hit_ratio"][0],
                        ret["hit_ratio"][-1],
                        ret["ndcg"][0],
                        ret["ndcg"][-1],
                    )
                )
                print(perf_str)
            # 현재의 에폭 recall값이 이전 epoc에서의 best recall보다 높은지 검사
            if ret["recall"][1] > best_recall:
                # 업데이트
                best_recall = ret["recall"][1]
                stopping_step = 0
                # 현재 모델 상태 저장
                torch.save(
                    {self.model_name: self.model.state_dict()},
                    "./models/" + self.dataset + "_" + self.model_name,
                )
            # 지정된 값에 아직 도달 안했으면
            elif stopping_step < args.early_stopping_patience:
                # step을 올림
                stopping_step += 1
                print("#####Early stopping steps: %d #####" % stopping_step)
            # 도달되면 조기종료함
            else:
                print("#####Early stop! #####")
                break
        # 학습 끝!

        # 4. 최종 모델 평가
        # 모델 초기화
        self.model = MONET(
            self.n_users,
            self.n_items,
            self.feat_embed_dim,
            self.nonzero_idx,
            self.has_norm,
            self.image_feats,
            self.text_feats,
            self.n_layers,
            self.alpha,
            self.beta,
            self.agg,
            self.cf,
            self.cf_gcn,
            self.lightgcn,
        )


        # 저장된 모달 상태 불러와서 적용
        self.model.load_state_dict(
            torch.load(
                "./models/" + self.dataset + "_" + self.model_name,
                map_location=torch.device("cpu"),
            )[self.model_name]
        )
        # gpu이동
        self.model#cuda()
        # test set에 대해 평가함
        test_ret = self.test(users_to_test, is_val=False)
        # 최종 지표
        print("Final ", test_ret)


def set_seed(seed):
    # 시드 설정 -> 같은 난수 생성되도록
    # 넘파이, 파이썬 랜덤모듈, 파이토치 cpu와 gpu에도 모두 시드 설정해줌
    # -> 같은 초기 조건에서 같은 결과 계속 얻을 수 있음
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    #torch.cuda.manual_seed_all(seed)  # gpu


'''
main 실행 블록 -> 실험 수행하는 부분
- commandline argument 파싱 
- 시드 설정
- 모델 train
'''
if __name__ == "__main__":
    ## 명령어 파싱해옴
    args = parse_args(True)
    # 시드 설정
    set_seed(args.seed)

    # config에 실험의 기본 설정 (딕셔너리)
    # 유저, 아이템수 받아와서 config에 저장 (유저, 아이템 수, nonzero index 를)
    config = dict()
    config["n_users"] = data_generator.n_users
    config["n_items"] = data_generator.n_items

    nonzero_idx = data_generator.nonzero_idx()
    config["nonzero_idx"] = nonzero_idx

    # train함
    trainer = Trainer(config, args)
    trainer.train()
