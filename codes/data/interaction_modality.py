import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten


# Movielens

# 최대 사용자 ID와 아이템 ID 계산
max_user_id = max([int(user) for user in user_item_interactions.keys()]) + 1
max_item_id = max([max(items) for items in user_item_interactions.values()]) + 1

# 아이템별 상호작용 One-hot 인코딩 생성
interaction_matrix = np.zeros((max_item_id, max_user_id))
for user, items in user_item_interactions.items():
    for item in items:
        interaction_matrix[item, int(user)] = 1

# Dense 임베딩을 생성하기 위한 Autoencoder 모델
model = Sequential([
    Dense(128, activation='relu', input_shape=(max_user_id,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(max_user_id, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy')

# 모델 훈련
model.fit(interaction_matrix, interaction_matrix, epochs=10, batch_size=32)

# Dense 임베딩 추출
item_embeddings_dense = model.predict(interaction_matrix)

# Dense 임베딩을 NumPy 배열로 저장
np.save('item_embeddings_dense.npy', item_embeddings_dense)


# Tiktok
max_user_id=9308