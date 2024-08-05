import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim import optimizer
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
from collections import Counter
import os
import face_recognition as fr
from tqdm import tqdm
import random
from preprocess_images import preprocess_images 

# 설정 관련 딕셔너리 (하이퍼파라미터)
CFG = {
    'IMG_SIZE': 128,  # 이미지 사이즈 128
    'EPOCHS': 100,    # 에포크
    'BATCH_SIZE': 16, # 배치사이즈
    'SEED': 1,        # 시드
}

# 장치 설정 (CUDA 사용 가능 여부에 따라 CPU 또는 GPU 선택)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Loading...')

# 시드 설정 (재현 가능성을 위해 시드 고정) / 재현성 보장, 랜덤 요소 제어
seed = 1
torch.manual_seed(seed)                     # CPU       
torch.cuda.manual_seed(seed)                # GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True       # 하드웨어 및 입력 크기에 따라 최적의 연산 알고리즘을 선택하게 하여 성능을 향상
np.random.seed(seed)                        # NUMPY

# 데이터셋과 모델 설정
class GenderDataset(Dataset):
    def __init__(self, image, label, train=True, transform=None):
        self.transform = transform # 이미지 변환(transform) 객체를 저장
        self.img_list = image      # 이미지 데이터를 리스트로 저장
        self.label_list = label    # 레이블 데이터를 리스트로 저장

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        label = self.label_list[idx]
        img = Image.fromarray(np.uint8(self.img_list[idx])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

# custom dataset 사용 transform 
train_transform = torchvision.transforms.Compose([
    transforms.Resize([CFG['IMG_SIZE'], CFG['IMG_SIZE']]),          # 각 이미지 같은 크기로 resize
    transforms.ToTensor(),                                          # 이미지를 텐서로 변환
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # 평균과 표준편차를 0.5로 정규화
])

test_transform = torchvision.transforms.Compose([
    transforms.Resize([CFG['IMG_SIZE'], CFG['IMG_SIZE']]),          # 각 이미지 같은 크기로 resize
    transforms.ToTensor(),                                          # 이미지를 텐서로 변환
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # 평균과 표준편차를 0.5로 정규화
])

Horizontal_transform = torchvision.transforms.Compose([
    transforms.Resize([CFG['IMG_SIZE'], CFG['IMG_SIZE']]),          # 각 이미지 같은 크기로 resize
    transforms.RandomHorizontalFlip(1.0),                           # Horizontal = 좌우반전
    transforms.ToTensor(),                                          # 이미지를 텐서로 변환
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # 평균과 표준편차를 0.5로 정규화
])

# 학습 데이터 경로 설정
train_path = sorted(glob.glob(r"C:\Users\admin\Desktop\Gender\original images\*.jpg"))

# 이미지 전처리 실행 (캐싱)
train_path2, face_list = preprocess_images(train_path)

# 파일명을 이용한 성별 추출 및 레이블 설정
gender_labels = []
for id in train_path2:
    file_name_with_extension = os.path.basename(id)
    name = file_name_with_extension.split('A')[0]  # 'A' 앞의 부분 추출
    if int(name) <= 7380:
        gender_labels.append(1)  # 여성
    else:
        gender_labels.append(0)  # 남성

# 데이터셋 섞기 및 분할
random.Random(19991006).shuffle(face_list)
random.Random(19991006).shuffle(train_path2)
random.Random(19991006).shuffle(gender_labels)

# 전체 데이터셋 80% 훈련, 20% 검증
train_img_list = face_list[:int(len(face_list) * 0.8)]
train_label_list = gender_labels[:int(len(gender_labels) * 0.8)]
valid_img_list = face_list[int(len(face_list) * 0.8):]
valid_label_list = gender_labels[int(len(gender_labels) * 0.8):]

# 학습 DataSet과 검증 DataSet 생성
train_dataset = GenderDataset(image=train_img_list, label=train_label_list, train=True, transform=train_transform)
valid_dataset = GenderDataset(image=valid_img_list, label=valid_label_list, train=False, transform=test_transform)

# 학습 DataLoader와 검증 DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=16, num_workers=2, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, num_workers=2, shuffle=False)

# ResNet 기반 성별 예측 모델 정의
class ResNetGenderModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetGenderModel, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        resnet.fc = nn.Sequential(
            nn.Linear(512, 512),  # 512개의 입력 노드와 512개의 출력 노드를 가지는 완전 연결 계층
            nn.ReLU(),            # ReLU 활성화 함수
            nn.Dropout(0.3),      # 과적합을 방지하기 위한 드롭아웃 레이어, 드롭아웃 확률 30%
            nn.Linear(512, num_classes), # 512개의 입력 노드와 2개의 출력 노드를 가지는 완전 연결 계층
            nn.Softmax(dim=1)     # 각 클래스에 대한 확률을 계산 소프트맥스 함수
        )
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)
    
# 모델 인스턴스 생성 및 장치 설정
model1 = ResNetGenderModel(num_classes=2)
model1.to(device)

# 손실 함수 및 옵티마이저 설정
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model1.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = None



