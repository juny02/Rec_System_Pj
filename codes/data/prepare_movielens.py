import pandas as pd
import cv2
import os
import subprocess
import torchaudio
import librosa
import tensorflow_hub as hub
import numpy as np
import torch
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sentence_transformers import SentenceTransformer
from pytube import YouTube
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from oauth2client.tools import argparser

print("import done")

# 영화 데이터 불러오기
file_path = 'Movielens/ml-100k/u.item'
column_names = ['movie_id', 'movie_title', 'release_date', 'video_release_date',
                'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                'Thriller', 'War', 'Western']

movie_data = pd.read_csv(file_path, sep='|', names=column_names, encoding='latin-1', usecols=['movie_id', 'movie_title'])
'''
' 크롤링 '
DEVELOPER_KEY='AIzaSyDcshuuI63Fpy8XgsIr4PDnFCXWoMoJtuM'
YOUTUBE_API_SERVICE_NAME='youtube'
YOUTUBE_API_VERSION='v3'
youtube=build(YOUTUBE_API_SERVICE_NAME,YOUTUBE_API_VERSION,developerKey=DEVELOPER_KEY)
def youtube_search(search_keyword):
    search_response = youtube.search().list(
        q=search_keyword,
        part='id,snippet',
        maxResults=1,
        type='video'
    ).execute()

    if search_response['items']:
        first_result = search_response['items'][0]
        video_id = first_result['id']['videoId']
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        video_description = first_result['snippet']['description']
        return video_url, video_description
    else:
        print("Link making Failed")
        return None, None

# 디렉토리 생성
if not os.path.exists('Movielens/videos'):
    os.makedirs('Movielens/videos')
if not os.path.exists('Movielens/audio'):
    os.makedirs('Movielens/audio')
if not os.path.exists('Movielens/text'):
    os.makedirs('Movielens/text')
# description 파일
description_file = open('Movielens/text/descriptions.txt', 'w', encoding='utf-8')

for index, row in movie_data.iterrows():
    movie_title = row['movie_title']
    movie_id = row['movie_id']

    search_keyword = movie_title + " trailer"

    video_url, video_description = youtube_search(search_keyword)

    if video_url:
        yt = YouTube(video_url)
        # 비디오 다운로드
        video_stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if video_stream:
            video_filename = f"{movie_id}.mp4"
            video_filepath = os.path.join('Movielens/videos', video_filename)
            video_stream.download(output_path='Movielens/videos', filename=video_filename)
            print(f"Downloaded video: {movie_title}")
        else:
            print("No video stream found for: {movie_title}")

        # audio 다운로드
        audio_stream = yt.streams.filter(only_audio=True).first()
        if audio_stream:
            audio_file_path = os.path.join('Movielens/audio', f"{movie_id}.mp3")
            audio_stream.download(output_path='Movielens/audio', filename=f"{movie_id}.mp3")
            print(f"Downloaded audio: {movie_title}")

            wav_file_path = audio_file_path.replace('.mp3', '.wav')
            subprocess.run([
                'ffmpeg',
                '-i', audio_file_path,  # 입력 오디오 파일 (MP3)
                '-acodec', 'pcm_s16le',  # PCM 16-bit 변환
                '-ar', '16000',  # 16kHz 샘플링 레이트
                '-ac', '1',  # 모노 채널
                wav_file_path  # 출력 오디오 파일 (WAV)
            ])
            print(f"Converted audio to WAV: {wav_file_path}")

        else:
            print(f"No audio stream found for: {movie_title}")
        os.remove(audio_file_path)

        # text 로드
        description_file.write(f"{movie_title} {video_description}\n")
    else:
        print(f"No results for: {movie_title}")

description_file.close()

'Image'
# 프레임 추출
def extract_frames(video_path, frames_dir, every_n_frames=60):
    if not os.path.exists(video_path):
        print(f"Video path does not exist: {video_path}")
        return
    os.makedirs(frames_dir, exist_ok=True)
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        if count % every_n_frames == 0:  # every_n_frames를 조절하여 키 프레임 추출 간격을 설정
            frame_path = os.path.join(frames_dir, f"frame_{count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
        count += 1
    video.release()

# 모델 불러오기
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
# 이미지를 로드하고 전처리하는 함수
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features

# 이미지 폴더 및 파일 설정
video_directory = 'Movielens/videos'
frames_directory = 'Movielens/frames'
features_directory = 'Movielens'

# 모든 비디오에 대해 프레임 추출 및 특성 추출
all_features = []
video_files = os.listdir(video_directory)
video_files.sort(key=lambda filename: int(os.path.splitext(filename)[0]))  # 파일 이름(숫자) 기준으로 정렬

for video_name in video_files:
    print(video_name)
    video_path = os.path.join(video_directory, video_name)
    base_name = os.path.splitext(video_name)[0]  # 확장자 제외
    video_frames_dir = os.path.join(frames_directory, base_name)
    video_features_dir = os.path.join(features_directory, base_name)
    #os.makedirs(video_features_dir, exist_ok=True)

    # 프레임 추출
    extract_frames(video_path, video_frames_dir, every_n_frames=60)  # 매 60 프레임마다 추출 (조정 가능)

    # 프레임별 특성 추출 및 저장
    video_features = []
    for frame_name in os.listdir(video_frames_dir):
        frame_path = os.path.join(video_frames_dir, frame_name)
        features = extract_features(frame_path, model)
        video_features.append(features)

    # 프레임 특성의 평균 계산
    if video_features:
        video_features = np.array(video_features)
        mean_features = np.mean(video_features, axis=0)
    else:
        mean_features = np.zeros((1, 2048))  # 빈 비디오의 경우 0 벡터로 처리
    all_features.append(mean_features)

# 모든 비디오 특성을 NumPy 배열로 변환하고 파일로 저장
all_features = np.array(all_features, dtype=np.float16).squeeze()  # 불필요한 차원 제거
np.save('Movielens/image_feat.npy', all_features)
'''

'Acoustic'
# VGGish 모델 불러오기
vggmodel = hub.load('https://tfhub.dev/google/vggish/1')

# 오디오 파일에서 임베딩 추출ㅛ
def embedding_from_fn(file_path):
    # 오디오 파일 로드 및 리샘플링
    audio, sr = librosa.load(file_path, sr=None)
    audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000) # 리샘플링하여 16kHz로 변경
    embeddings = np.array(vggmodel(audio_16k))
    mean_embedding = np.mean(embeddings, axis=0)
    return mean_embedding


# 오디오 파일 경로
audio_directory = 'Movielens/audio'
acoustic_feats = []

# 오디오 파일들 처리
audio_files = os.listdir(audio_directory)
audio_files.sort(key=lambda filename: int(os.path.splitext(filename)[0]))  # 파일 이름(숫자) 기준으로 정렬

for audio_file in audio_files:
    audio_path = os.path.join(audio_directory, audio_file)

    # 임베딩 추출
    embedding = embedding_from_fn(audio_path)

    # 추출된 임베딩을 배열에 추가
    acoustic_feats.append(embedding)

# 모든 오디오 특성을 하나의 NumPy 배열로 변환하고 파일로 저장
acoustic_feats = np.array(acoustic_feats, dtype=np.float16)
np.save('Movielens/acoustic_feats.npy', acoustic_feats)
'''


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

acoustic_feats = []
audio_directory = 'Movielens/audio'

for audio_file in os.listdir(audio_directory):
    audio_path = os.path.join(audio_directory, audio_file)

    # 오디오 파일 로드 (torchaudio 사용)
    waveform, sample_rate = torchaudio.load(audio_path)

    # 오디오 파일을 모델의 샘플링 레이트로 리샘플링
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform).squeeze()

    # 오디오 파일을 토크나이저로 전처리하고 모델 입력 형식으로 변환
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)

    # wav2vec 2.0 모델을 사용하여 특성 추출
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state

    # 특성을 NumPy 배열로 변환
    acoustic_feats.append(hidden_states.squeeze().cuda().numpy())

# 모든 오디오 특성을 하나의 NumPy 배열로 변환하고 파일로 저장
acoustic_feats = np.array(acoustic_feats, dtype=np.float32)
np.save('Movielens/acoustic_feats.npy', acoustic_feats)
'''

'text'
# text feature 처리
# 버트 기반 언어 모델 로드
bert_path = "sentence-transformers/stsb-roberta-large"
bert_model = SentenceTransformer(bert_path)

t_features = []

# descriptions.txt 파일 읽기
with open('Movielens/text/descriptions.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 각 줄에 대해 반복
for line in lines:
    # 공백 줄이 아닌 경우에만 처리
    if line.strip():
        embedding = bert_model.encode(line.strip())
        t_features.append(embedding)

# 배열로 변환 후 저장
t_features = np.asarray(t_features, dtype=np.float16)
np.save('Movielens/text_feat.npy', t_features)