import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
# mac
device = torch.device("mps")

'''
normalized laplacian을 얻는 함수
- edge_index: 그래프의 엣지 정보 나타내는 텐서, 출발노드 도착노드로 구성
- edge_weight: 각 엣지에 대한 가중치 -> 이 함수로 업데이트 함
'''
def normalize_laplacian(edge_index, edge_weight):
    # 그래프의 노드 개수를 얻음
    num_nodes = maybe_num_nodes(edge_index)

    # 출발노드, 도착노드 추출
    row, col = edge_index[0], edge_index[1]

    #각 노드의 차수 나타내는 텐서 만듦 (행렬)
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    # 차수에 -0.5 제곱 씌움 (루트역수)
    deg_inv_sqrt = deg.pow_(-0.5)

    # 무한대값 0으로 (역제곱근 씌우니까 처리해줌)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)

    # 1/루트 출발노드 x 루트 도착노드 (각 아이템, 유저의 one-hop 이웃 개수 루트곱 역수) 구함
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    # weight담은 행렬이 됨
    return edge_weight

'''
GCNs 구현 함수
- MessagePassing 클래스를 상속해서 구현
- propagate시 messange-aggregate-update 호출해서 전파하게 됨
- linear GCN (그냥 합으로 결합)
https://velog.io/@mesrwi/Pytorch-Geometric-Message-Passing-Network-iqir104j
'''
class Our_GCNs(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Our_GCNs, self).__init__(aggr="add") # 메세지 집계 - add
        self.in_channels = in_channels # 입력 특성 차원 수
        self.out_channels = out_channels # 출력 특성 차원 수

    def forward(self, x, edge_index, weight_vector, size=None):
        self.weight_vector = weight_vector
        return self.propagate(edge_index, size=size, x=x)

    # 이웃노드 정보
    def message(self, x_j):
        return x_j * self.weight_vector
    # aggregate (add)
    def update(self, aggr_out):
        return aggr_out


from torch_geometric.nn.inits import uniform

'''
Nonlinear GCN
- 노드들에 가중치 행렬을 곱해줌 
'''
class Nonlinear_GCNs(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Nonlinear_GCNs, self).__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 가중치 초기화 (학습 가능 parameter)
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self): #uniform 분포 활용
        uniform(self.in_channels, self.weight)

    # 메세지 전파시
    def forward(self, x, edge_index, weight_vector, size=None):
        # 가중치 곱
        x = torch.matmul(x, self.weight)
        self.weight_vector = weight_vector
        return self.propagate(edge_index, size=size, x=x)

    def message(self, x_j):
        return x_j * self.weight_vector

    def update(self, aggr_out):
        return aggr_out

'''
MeGCN
- linear
- self connection
'''
class MeGCN(nn.Module):
    def __init__(
        self,
        # 유저, 아이템 수
        n_users,
        n_items,
        # GCN 레이어 수
        n_layers,
        # feature 임베딩 정규화 여부
        has_norm,
        # 임베딩 차원 크기
        feat_embed_dim,
        # 그래프 엣지
        nonzero_idx,
        # 각 모달리티 사용 여부
        use_image,
        use_txt,
        use_acoustic,
        use_interact,
        #active_modal_num,
        # 각 모달리티 feature
        image_feats,
        text_feats,
        acoustic_feats,
        #interaction_feats,
        # 모달리티 feature 반영 정도 (self connection
        alpha,
        # aggregate 방법 - 최종 text, image 임베딩 합치는 방식
        agg,
        # cf 사용 여부
        cf,
        # cf의 gcn방법 - lightGCN or MeGCN..
        cf_gcn,
        # lightGCN 사용 여부
        lightgcn,
    ):
        # 초기화
        super(MeGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.has_norm = has_norm
        self.feat_embed_dim = feat_embed_dim
        self.nonzero_idx = torch.tensor(nonzero_idx).long().T#.cuda().long().T
        self.alpha = alpha
        self.agg = agg
        self.cf = cf
        self.cf_gcn = cf_gcn
        self.lightgcn = lightgcn
        self.use_image = use_image
        self.use_txt = use_txt
        self.use_acoustic = use_acoustic
        self.use_interact = use_interact
        self.active_modal_num = sum([
            use_image,
            use_txt,
            use_acoustic,
            use_interact
        ])


        # 임베딩들 초기화
        # 각 모달리티에서의 유저임베딩 - xavier로 초기화
        # 아이템 임베딩 - pretrained feature 임베딩 사용, 차원 축소
        if self.use_image:
            # 유저 임베딩
            self.image_preference = nn.Embedding(self.n_users, self.feat_embed_dim)
            nn.init.xavier_uniform_(self.image_preference.weight)
            # 아이템 임베딩
            self.image_embedding = nn.Embedding.from_pretrained(
                torch.tensor(image_feats, dtype=torch.float), freeze=True
            )  # [# of items, 4096]
            self.image_trs = nn.Linear(image_feats.shape[1], self.feat_embed_dim)

        if self.use_txt:
            self.text_preference = nn.Embedding(self.n_users, self.feat_embed_dim)
            nn.init.xavier_uniform_(self.text_preference.weight)
            self.text_embedding = nn.Embedding.from_pretrained(
                torch.tensor(text_feats, dtype=torch.float), freeze=True) # [# of items, 1024]
            self.text_trs = nn.Linear(text_feats.shape[1], self.feat_embed_dim)

        if self.use_acoustic:
            self.acoustic_preference = nn.Embedding(self.n_users, self.feat_embed_dim)
            nn.init.xavier_uniform_(self.acoustic_preference.weight)
            self.acoustic_embedding = nn.Embedding.from_pretrained(
                torch.tensor(acoustic_feats, dtype=torch.float), freeze=True)
            self.acoustic_trs = nn.Linear(acoustic_feats.shape[1], self.feat_embed_dim)


        if self.use_interact:
            self.interaction_preference = nn.Embedding(self.n_users, self.feat_embed_dim)
            nn.init.xavier_uniform_(self.interaction_preference.weight)
            self.interaction_embedding = nn.Embedding(self.n_items, self.feat_embed_dim)
            nn.init.xavier_uniform_(self.interaction_embedding.weight)


        # cf를 활용한다고 하면 유저, 아이템 임베딩 추가로 만들어줌 (cf에서 쓰는)
        if self.cf:
            self.user_embedding = nn.Embedding(self.n_users, self.feat_embed_dim)
            self.item_embedding = nn.Embedding(self.n_items, self.feat_embed_dim)
            nn.init.xavier_uniform_(self.user_embedding.weight)
            nn.init.xavier_uniform_(self.item_embedding.weight)

        # cf를 사용하지 않으면
        if not self.cf:
            # aggregator로 fc쓰면 다 잇고 다시 임베딩 차원으로 변환
            if self.agg == "fc":
                self.transform = nn.Linear(self.feat_embed_dim * self.active_modal_num, self.feat_embed_dim)
            # aggregator로 weighted sum사용 (더하는거)
            elif self.agg == "weighted_sum":
                self.modal_weight = nn.Parameter(torch.Tensor([1.0 / self.active_modal_num] * self.active_modal_num))
                self.softmax = nn.Softmax(dim=0)

        # cf를 사용하면 각 경우에서 이제 cf정보도 포함해야하니까 추가해주면 됨
        else:
            if self.agg == "fc": #3개 잇게 됨
                self.transform = nn.Linear(self.feat_embed_dim * self.active_modal_num+1, self.feat_embed_dim)
            elif self.agg == "weighted_sum": # 3개 합침
                self.modal_weight = nn.Parameter(torch.Tensor([1.0 / self.active_modal_num+1] * self.active_modal_num))
                self.softmax = nn.Softmax(dim=0)

        # Our_GCN 모듈 배열
        # 각 갯수만큼 적용한 레이어들 있음
        self.layers = nn.ModuleList(
            [
                Our_GCNs(self.feat_embed_dim, self.feat_embed_dim)
                for _ in range(self.n_layers)
            ]
        )

    def forward(self, edge_index, edge_weight, _eval=False):
        # 1. 임베딩
        # 전체를 배열에 저장
        ego_embeddings = []
        all_embeddings_list = []
        final_preferences = []
        final_embeddings = []

        if self.use_image:
            # transform - image, text embedding의 선형변환 -> 차원 맞춰줌
            image_emb = self.image_trs(self.image_embedding.weight) # [# of items, feat_embed_dim]
            if self.has_norm:
                image_emb = F.normalize(image_emb)
            # image, text preference 임베딩 (모달리티별 유저임베딩)
            image_preference = self.image_preference.weight
            # 유저, 아이템 임베딩을 이어줌 (유저-아이템 수직)
            ego_image_emb = torch.cat([image_preference, image_emb], dim=0)
            ego_embeddings.append(ego_image_emb)
            if self.lightgcn: # (이 방식은 마지막에 종합하므로 레이어별로 얻은 값 다 필요함)
                all_image_emb = [ego_image_emb]
                all_embeddings_list.append(all_image_emb)


        if self.use_txt:
            text_emb = self.text_trs(self.text_embedding.weight)
            if self.has_norm:
                text_emb = F.normalize(text_emb)
            text_preference = self.text_preference.weight
            ego_text_emb = torch.cat([text_preference, text_emb], dim=0)
            ego_embeddings.append(ego_text_emb)
            if self.lightgcn:
                all_text_emb = [ego_text_emb]
                all_embeddings_list.append(all_text_emb)

        if self.use_acoustic:
            acoustic_emb = self.acoustic_trs(self.acoustic_embedding.weight)
            if self.has_norm:
                acoustic_emb = F.normalize(acoustic_emb)
            acoustic_preference = self.acoustic_preference.weight
            ego_acoustic_emb = torch.cat([acoustic_preference, acoustic_emb], dim=0)
            ego_embeddings.append(ego_acoustic_emb)
            if self.lightgcn:
                all_acoustic_emb = [ego_acoustic_emb]
                all_embeddings_list.append(all_acoustic_emb)

        if self.use_interact:
            interaction_emb = self.interaction_embedding.weight
            if self.has_norm:
                interaction_emb = F.normalize(interaction_emb)
            interaction_preference = self.interaction_preference.weight
            ego_interaction_emb = torch.cat([interaction_preference, interaction_emb], dim=0)
            ego_embeddings.append(ego_interaction_emb)
            if self.lightgcn:
                all_interaction_emb = [ego_interaction_emb]
                all_embeddings_list.append(all_interaction_emb)


        # cf도 추가적으로 사용한다면 그것도 추가로 만들어줌 - ego만들어줌
        if self.cf:
            user_emb = self.user_embedding.weight
            item_emb = self.item_embedding.weight
            ego_cf_emb = torch.cat([user_emb, item_emb], dim=0)
            if self.cf_gcn == "LightGCN": # lightGCN 으로 하면 다음처럼 넣어줌
                all_cf_emb = [ego_cf_emb]

        # 2. GCN 레이어 적용

        # 모든 레이어에 대해
        for layer in self.layers:
            if not self.lightgcn:  # lightGCN아닌 MeGCN한다면
                if self.use_image:
                    # GCN 적용해서 얻어진 임베딩 (linear 로 이웃노드 결합된 임베딩 값)
                    side_image_emb = layer(ego_image_emb, edge_index, edge_weight)
                    # 해당 값에 alpha 만큼의 self-connection 추가해줌
                    ego_image_emb = side_image_emb + self.alpha * ego_image_emb
                if self.use_txt:
                    side_text_emb = layer(ego_text_emb, edge_index, edge_weight)
                    ego_text_emb = side_text_emb + self.alpha * ego_text_emb
                if self.use_acoustic:
                    side_acoustic_emb = layer(ego_acoustic_emb, edge_index, edge_weight)
                    ego_acoustic_emb = side_acoustic_emb + self.alpha * ego_acoustic_emb
                if self.use_interact:
                    side_interaction_emb = layer(ego_interaction_emb, edge_index, edge_weight)
                    ego_interaction_emb = side_interaction_emb + self.alpha * ego_interaction_emb

            else:  # LightGCN방식이면 -> 이웃합만 집계하게 됨
                if self.use_image:
                    side_image_emb = layer(ego_image_emb, edge_index, edge_weight)
                    ego_image_emb = side_image_emb
                    all_image_emb += [ego_image_emb]

                if self.use_txt:
                    side_text_emb = layer(ego_text_emb, edge_index, edge_weight)
                    ego_text_emb = side_text_emb
                    all_text_emb += [ego_text_emb]

                if self.use_acoustic:
                    side_acoustic_emb = layer(ego_acoustic_emb, edge_index, edge_weight)
                    ego_acoustic_emb = side_acoustic_emb
                    all_acoustic_emb += [ego_acoustic_emb]

                if self.use_interaction:
                    side_interaction_emb = layer(ego_interaction_emb, edge_index, edge_weight)
                    ego_interaction_emb = side_interaction_emb
                    all_interaction_emb += [ego_interaction_emb]
        '''
        for layer in self.layers:
            if not self.lightgcn:  # MeGCN 방식
                for i, ego_emb in enumerate(ego_embeddings):
                    side_emb = layer(ego_emb, edge_index, edge_weight)
                    # self-connection 추가
                    ego_embeddings[i] = side_emb + self.alpha * ego_emb

            else:  # LightGCN 방식
                for i, ego_emb in enumerate(ego_embeddings):
                    side_emb = layer(ego_emb, edge_index, edge_weight)
                    ego_embeddings[i] = side_emb  # 이웃합만 집계
                    all_embeddings_list[i] += [ego_embeddings[i]]  # 각 모달리티별 임베딩 추가
        '''

        # 3. GCN 에서 최종 모달리티별 유저, 아이템 임베딩 얻음 (preference, emb)
        if not self.lightgcn: #MeGCN이면 각각 유저, 아이템 임베딩 split해서 저장해줌
            if self.use_image:
                final_image_preference, final_image_emb = torch.split(
                    ego_image_emb, [self.n_users, self.n_items], dim=0
                )
            if self.use_txt:
                final_text_preference, final_text_emb = torch.split(
                    ego_text_emb, [self.n_users, self.n_items], dim=0
                )
            if self.use_acoustic:
                final_acoustic_preference, final_acoustic_emb = torch.split(
                    ego_acoustic_emb, [self.n_users, self.n_items], dim=0
                )
            if self.use_interact:
                final_interaction_preference, final_interaction_emb = torch.split(
                    ego_interaction_emb, [self.n_users, self.n_items], dim=0
                )

        else: #LightGCN이면 각 모달리티에서 GCN통해 얻은 모든 레이어에서의 임베딩을 평균내서 사용. stack해서 평균내줌
            if self.use_image:
                all_image_emb = torch.stack(all_image_emb, dim=1)
                all_image_emb = all_image_emb.mean(dim=1, keepdim=False)
                final_image_preference, final_image_emb = torch.split(
                    all_image_emb, [self.n_users, self.n_items], dim=0
                )
            if self.use_txt:
                all_text_emb = torch.stack(all_text_emb, dim=1)
                all_text_emb = all_text_emb.mean(dim=1, keepdim=False)
                final_text_preference, final_text_emb = torch.split(
                    all_text_emb, [self.n_users, self.n_items], dim=0
                )
            if self.use_acoustic:
                all_acoustic_emb = torch.stack(all_acoustic_emb, dim=1)
                all_acoustic_emb = all_acoustic_emb.mean(dim=1, keepdim=False)
                final_acoustic_preference, final_acoustic_emb = torch.split(
                    all_acoustic_emb, [self.n_users, self.n_items], dim=0
                )
            if self.use_interaction:
                all_interaction_emb = torch.stack(all_interaction_emb, dim=1)
                all_interaction_emb = all_interaction_emb.mean(dim=1, keepdim=False)
                final_interaction_preference, final_interaction_emb = torch.split(
                    all_interaction_emb, [self.n_users, self.n_items], dim=0
                )

        # cf를 사용했으면 그 final 임베딩도 각각 구해줌
        if self.cf:
            if self.cf_gcn == "MeGCN": # 그냥 나눔
                final_cf_user_emb, final_cf_item_emb = torch.split(
                    ego_cf_emb, [self.n_users, self.n_items], dim=0
                )
            elif self.cf_gcn == "LightGCN": # 누적 평균냄
                all_cf_emb = torch.stack(all_cf_emb, dim=1)
                all_cf_emb = all_cf_emb.mean(dim=1, keepdim=False)
                final_cf_user_emb, final_cf_item_emb = torch.split(
                    all_cf_emb, [self.n_users, self.n_items], dim=0
                )

        # 평가모드일 경우 final 유저,아이템 임베딩 모두 반환
        if _eval:
            return ego_image_emb, ego_text_emb, ego_acoustic_emb, ego_interaction_emb

        # 4. fused embedding 생성 (final embedding !)
        # aggregation 방식에 따라 각 모달리티별로  final 아이템, 유저 임베딩을 합쳐줌
        # a. 전체 모달리티 임베딩 합쳐줌
        item_embeddings = []
        user_preference_embeddings = []

        if self.use_image:
            item_embeddings.append(final_image_emb)
            user_preference_embeddings.append(final_image_preference)

        if self.use_txt:
            item_embeddings.append(final_text_emb)
            user_preference_embeddings.append(final_text_preference)

        if self.use_acoustic:
            item_embeddings.append(final_acoustic_emb)
            user_preference_embeddings.append(final_acoustic_preference)

        if self.use_interact:
            item_embeddings.append(final_interaction_emb)
            user_preference_embeddings.append(final_interaction_preference)

        if not self.cf:  # cf를 사용하지 않을때 (..)
            if self.agg == "concat":
                # 아이템 임베딩
                items = torch.cat(item_embeddings, dim=1)
                # 유저 임베딩
                user_preference = torch.cat(user_preference_embeddings, dim=1)
            elif self.agg == "sum": # 합해서 구함
                # 아이템 임베딩
                items = sum(item_embeddings)
                # 유저 임베딩
                user_preference = sum(user_preference_embeddings)
            elif self.agg == "weighted_sum":
                weight = self.softmax(self.modal_weight)
                items = sum(w * emb for w, emb in zip(weight, item_embeddings))
                user_preference = sum(w * emb for w, emb in zip(weight, user_preference_embeddings))
            elif self.agg == "fc": # concate하고 feat_embed_dim차원으로
                items = self.transform(torch.cat(item_embeddings, dim=1))
                user_preference = self.transform(torch.cat(user_preference_embeddings, dim=1))

        # *** 사용안할거라서 수정 아직 안함 ***
        # cf를 사용하면 합칠때 cf로 얻은 임베딩도 같이 합쳐주면 됨
        # 위와 동일하고 거기에 final_cf_item_emb, final_cf_user_emb이 추가됨
        else:
            if self.agg == "concat":
                items = torch.cat(
                    [final_image_emb, final_text_emb, final_cf_item_emb], dim=1
                )  # [# of items, feat_embed_dim * 2]
                user_preference = torch.cat(
                    [final_image_preference, final_text_preference, final_cf_user_emb],
                    dim=1,
                )  # [# of users, feat_embed_dim * 2]
            elif self.agg == "sum":
                items = (
                    final_image_emb + final_text_emb + final_cf_item_emb
                )  # [# of items, feat_embed_dim]
                user_preference = (
                    final_image_preference + final_text_preference + final_cf_user_emb
                )  # [# of users, feat_embed_dim]
            elif self.agg == "weighted_sum":
                weight = self.softmax(self.modal_weight)
                items = (
                    weight[0] * final_image_emb
                    + weight[1] * final_text_emb
                    + weight[2] * final_cf_item_emb
                )  # [# of items, feat_embed_dim]
                user_preference = (
                    weight[0] * final_image_preference
                    + weight[1] * final_text_preference
                    + weight[2] * final_cf_user_emb
                )  # [# of users, feat_embed_dim]
            elif self.agg == "fc":
                items = self.transform(
                    torch.cat(
                        [final_image_emb, final_text_emb, final_cf_item_emb], dim=1
                    )
                )  # [# of items, feat_embed_dim]
                user_preference = self.transform(
                    torch.cat(
                        [
                            final_image_preference,
                            final_text_preference,
                            final_cf_user_emb,
                        ],
                        dim=1,
                    )
                )  # [# of users, feat_embed_dim]

        # 최종 아이템, 유저 임베딩 반환
        return user_preference, items


class MONET(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        feat_embed_dim,
        nonzero_idx,
        has_norm,
        image_feats,
        text_feats,
        acoustic_feats,
        #interaction_feats,
        n_layers,
        alpha,
        beta, #matching score 점수 조정
        agg,
        cf,
        cf_gcn,
        lightgcn,
        use_image,
        use_txt,
        use_acoustic,
        use_interact,
    ):
        super(MONET, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.feat_embed_dim = feat_embed_dim
        self.n_layers = n_layers
        self.nonzero_idx = nonzero_idx
        self.alpha = alpha
        self.beta = beta
        self.agg = agg
        self.image_feats = torch.tensor(image_feats, dtype=torch.float)#.cuda()
        self.text_feats = torch.tensor(text_feats, dtype=torch.float)#.cuda()
        self.acoustic_feats = torch.tensor(acoustic_feats, dtype=torch.float)#.cuda()
        #self.interaction_feats = torch.tensor(interaction_feats, dtype=torch.float).cuda()
        self.use_image = use_image
        self.use_txt = use_txt
        self.use_acoustic = use_acoustic
        self.use_interact = use_interact

        # MeGCN 모듈 사용하므로
        self.megcn = MeGCN(
            self.n_users,
            self.n_items,
            self.n_layers,
            has_norm,
            self.feat_embed_dim,
            self.nonzero_idx,
            use_image,
            use_txt,
            use_acoustic,
            use_interact,
            image_feats,
            text_feats,
            acoustic_feats,
            #interaction_feats,
            self.alpha,
            self.agg,
            cf,
            cf_gcn,
            lightgcn,
        )
        ''' [ 그래프 구성 ] '''
        nonzero_idx = torch.tensor(self.nonzero_idx).long().T#.cuda().long().T

        # 유저 노드로 구분
        nonzero_idx[1] = nonzero_idx[1] + self.n_users

        # 역방향 엣지 쌓아서 edge_index 만듦 (이어붙임)
        self.edge_index = torch.cat(
            [nonzero_idx, torch.stack([nonzero_idx[1], nonzero_idx[0]], dim=0)], dim=1
        )

        # 전체 edge에 대해 1로 weight초기화 (1차원)
        self.edge_weight = torch.ones((self.edge_index.size(1))).view(-1,1)#.cuda().view(-1, 1)

        # 라플라시안 적용 -> weight 업데이트
        self.edge_weight = normalize_laplacian(self.edge_index, self.edge_weight)

        # 다시 tensor로 변환
        nonzero_idx = torch.tensor(self.nonzero_idx).long().T#.cuda().long().T

        # 인접행렬 생성 - uxi 행렬에서 비어있지 않은 원소는 1로 초기화
        self.adj = (
            torch.sparse.FloatTensor(
                nonzero_idx,
                torch.ones((nonzero_idx.size(1))),#.cuda(),
                (self.n_users, self.n_items),
            )
            .to_dense()
            #cuda()
        )

    def forward(self, _eval=False):
        if _eval: # 테스트 모드일시 합쳐진 최종 user, item 임베딩이 아닌 각 모달리티별의 유저,모달리티 임베딩 모두 얻게됨
            img, txt = self.megcn(self.edge_index, self.edge_weight, _eval=True)
            return img, txt

        # 최종 유저, 아이템 임베딩
        user, items = self.megcn(self.edge_index, self.edge_weight, _eval=False)

        return user, items

    # 최종 preference 및 loss 구함
    def bpr_loss(self, user_emb, item_emb, users, pos_items, neg_items, target_aware):
        current_user_emb = user_emb[users]
        # 평가한 아이템
        pos_item_emb = item_emb[pos_items]
        # 평가 x 아이템
        neg_item_emb = item_emb[neg_items]

        '''target-aware'''
        if target_aware:
            # target-aware
            item_item = torch.mm(item_emb, item_emb.T)
            pos_item_query = item_item[pos_items, :]  # (batch_size, n_items)
            neg_item_query = item_item[neg_items, :]  # (batch_size, n_items)
            pos_target_user_alpha = torch.softmax(
                torch.multiply(pos_item_query, self.adj[users, :]).masked_fill(
                    self.adj[users, :] == 0, -1e9
                ),
                dim=1,
            )  # (batch_size, n_items)
            neg_target_user_alpha = torch.softmax(
                torch.multiply(neg_item_query, self.adj[users, :]).masked_fill(
                    self.adj[users, :] == 0, -1e9
                ),
                dim=1,
            )  # (batch_size, n_items)
            pos_target_user = torch.mm(
                pos_target_user_alpha, item_emb
            )  # (batch_size, dim)
            neg_target_user = torch.mm(
                neg_target_user_alpha, item_emb
            )  # (batch_size, dim)

            '''predictor'''
            # beta 값으로 target-aware 비율 조정
            pos_scores = (1 - self.beta) * torch.sum(
                torch.mul(current_user_emb, pos_item_emb), dim=1
            ) + self.beta * torch.sum(torch.mul(pos_target_user, pos_item_emb), dim=1)
            neg_scores = (1 - self.beta) * torch.sum(
                torch.mul(current_user_emb, neg_item_emb), dim=1
            ) + self.beta * torch.sum(torch.mul(neg_target_user, neg_item_emb), dim=1)

        # target-aware 사용 안할시
        else:
            pos_scores = torch.sum(torch.mul(current_user_emb, pos_item_emb), dim=1)
            neg_scores = torch.sum(torch.mul(current_user_emb, neg_item_emb), dim=1)

        # log 시그모이드 함수, -평균 적용 -> 최종 mf poss
        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)
        # 정규화항
        regularizer = (
            1.0 / 2 * (pos_item_emb**2).sum()
            + 1.0 / 2 * (neg_item_emb**2).sum()
            + 1.0 / 2 * (current_user_emb**2).sum()
        )
        #임베딩 크기로 나눠줌
        emb_loss = regularizer / pos_item_emb.size(0)

        reg_loss = 0.0

        return mf_loss, emb_loss, reg_loss

# bpr