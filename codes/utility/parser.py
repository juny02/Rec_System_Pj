import argparse

'''
Parser !
-> argparse 모듈 사용해서 commandline argument 분석, 처리
- 다양한 설정들을 입력받기 위함
- 입력 예시
    - python main.py --agg=concat --n_layers=0 --alpha=1.0 --beta=0.3 --dataset=WomenClothing --model_name=MONET_wo_MeGCN
    - python main.py --target_aware --agg=concat --n_layers=2 --alpha=1.0 --beta=0.3 --dataset=WomenClothing --model_name=MONET_wo_TA
    - python script.py --dataset MenClothing --epoch 100 --lr 0.001 
'''



def parse_args(flags=False):
    # argparse.ArgumentParser 인스턴스 생성
    parser = argparse.ArgumentParser(description="")

    '''
    입력하는 argument들 정의
    - --data_path "path/path"  다음처럼 주게됨
    '''
    # 데이터 경로
    parser.add_argument(
        "--data_path", nargs="?", default="data/", help="Input data path."
    )
    # seed값
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    # dataset지정
    parser.add_argument(
        "--dataset",
        nargs="?",
        default="MenClothing",
        help="Choose a dataset from {Toys_and_Games, Beauty, MenClothing, WomenClothing}",
    )
    # verbose (평가 주기)
    parser.add_argument(
        "--verbose", type=int, default=5, help="Interval of evaluation."
    )
    # 에폭
    parser.add_argument("--epoch", type=int, default=1000, help="Number of epoch.")
    # batch size
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size.")
    # 정규화 매개변수
    parser.add_argument(
        "--regs", nargs="?", default="[1e-5,1e-5]", help="Regularizations."
    )
    # learning rate
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    # 임베딩 크기
    parser.add_argument("--embed_size", type=int, default=64, help="Embedding size.")
    # feature 임베딩 크기
    parser.add_argument(
        "--feat_embed_dim", type=int, default=64, help="Feature embedding size."
    )
    # alpha
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Coefficient of self node features."
    )
    # beta
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="Coefficient of fine-grained interest matching.",
    )
    # warmstart, cold start
    parser.add_argument(
        "--core",
        type=int,
        default=5,
        help="5-core for warm-start; 0-core for cold start.",
    )
    # GCN 레이어 갯수
    parser.add_argument(
        "--n_layers", type=int, default=2, help="Number of graph conv layers."
    )
    # 정규화 할지 여부
    parser.add_argument("--has_norm", default=True, action="store_false")
    # target-aware여부
    parser.add_argument("--target_aware", default=True, action="store_false")
    # aggregate 방식
    parser.add_argument(
        "--agg",
        type=str,
        default="concat",
        help="Choose a dataset from {sum, weighted_sum, concat, fc}",
    )
    # cf 추가 사용 할지
    parser.add_argument("--cf", default=False, action="store_true")
    # cf의 GCN 방식 지정
    parser.add_argument(
        "--cf_gcn",
        type=str,
        default="LightGCN",
        help="Choose a dataset from {MeGCN, LightGCN}",
    )
    # lightGCN 사용할지 말지
    parser.add_argument("--lightgcn", default=False, action="store_true")
    # 모델 이름 지정
    parser.add_argument("--model_name", type=str)
    # 조기 종료 설정값
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="")
    # 사용할 gpu id
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id")
    # K값들
    parser.add_argument(
        "--Ks", nargs="?", default="[10, 20]", help="K value of ndcg/recall @ k"
    )
    # Add arguments for modality usage
    parser.add_argument(
        "--use_image", default=False, action="store_true",
        help="Use image modality if this flag is set."
    )
    parser.add_argument(
        "--use_txt", default=False, action="store_true",
        help="Use text modality if this flag is set."
    )
    parser.add_argument(
        "--use_acoustic", default=False, action="store_true",
        help="Use acoustic modality if this flag is set."
    )
    parser.add_argument(
        "--use_interact", default=False, action="store_true",
        help="Use interaction modality if this flag is set."
    )




    # test유형 - 미니배치 여부
    parser.add_argument(
        "--test_flag",
        nargs="?",
        default="part",
        help="Specify the test type from {part, full}, indicating whether the reference is done in mini-batch",
    )

    # flag가 true이면 실험 설정 출력함
    if flags:
        attribute_dict = dict(vars(parser.parse_args()))
        print("*" * 32 + " Experiment setting " + "*" * 32)
        for k, v in attribute_dict.items():
            print(k + " : " + str(v))
        print("*" * 32 + " Experiment setting " + "*" * 32)

    # 파싱된 argument 반환
    return parser.parse_args()
