import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models.convnet import *  # 모델 임포트
from tqdm import tqdm
import os
from PIL import Image
from matplotlib import pyplot as plt
import platform  # 폰트관련 운영체제 확인

# 커스텀 데이터셋 클래스 정의
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, dataset_type, transform=None):
        self.data_dir = os.path.join(data_dir, dataset_type)  # dataset_type (train/test) 폴더 경로
        self.transform = transform
        
        # 클래스별 폴더 목록 가져오기
        self.classes = sorted(os.listdir(self.data_dir))
        
        # 이미지 파일과 라벨 리스트 생성
        self.image_files = []
        self.labels = []
        
        # 각 클래스 폴더에서 이미지 파일 수집
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.data_dir, class_name)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(class_name, img_name))
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def main():
    # plot용 한글 폰트 설정
    if platform.system() == 'Windows':
        font_name = 'Malgun Gothic'
    elif platform.system() == 'Darwin':
        font_name = 'AppleGothic'
    else:
        font_name = 'NanumGothic'
        
    plt.rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False

    # GPU 사용 가능 여부 확인 및 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu') # cuda 사용시 blue screen 발생
    
    print(f'사용 중인 장치: {device}')
    
    ################################################
    # 하이퍼파라미터 설정    
    input_height = 10   # ConvNet 입력 size
    input_width = 28    # ConvNet 입력 size

    num_epochs = 10
    batch_size = 64
    learning_rate = 0.0001
    accumulation_steps = 1  # 그래디언트 누적 스텝 수
    ################################################
    
    if device.type == 'cuda':        
        # CUDA 설정
        torch.backends.cudnn.benchmark = True # 속도 향상을 위한 설정
        torch.backends.cudnn.deterministic = True # 재현 가능성 확보
        torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat-32 사용        
        torch.cuda.empty_cache()
        # 메모리 할당 모드 설정
        torch.cuda.set_per_process_memory_fraction(0.8)  # GPU 메모리의 80% 사용

    # 데이터 전처리 및 데이터로더 설정
    transform_train = transforms.Compose([
        transforms.Resize((input_height, input_width)),
        # transforms.Grayscale(num_output_channels=3),  # 3채널로 변환
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 예시로 3 채널 이미지에 대한 정규화
    ])

    train_dataset = CustomDataset(
        # data_dir='./data/transfer_learn',
        data_dir='./data',
        dataset_type='train',
        transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )

    # 모델 초기화 전에 클래스 수 확인 및 출력
    num_classes = len(train_dataset.classes)  # classes 리스트의 길이로 계산
    print(f'감지된 클래스 수: {num_classes}')
    
    # pre-training된 모델 경로 설정
    pretrained_model_path = "trained_model.pth"
        
    #################################################################
    # 모델 초기화
    model = ConvNet(num_classes, input_height, input_width).to(device)  # 모델을 장치로 이동
    
    # 기존 모델 가중치 로드 (학습된 모델의 경로)
    pretrained_dict = torch.load(pretrained_model_path, weights_only=True)
    model_dict = model.state_dict()

    # 기존 가중치에서 fc1, fc2, fc3를 제외한 나머지 가중치만 업데이트
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc3' not in k}
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc1' not in k and 'fc2' not in k and 'fc3' not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False) # strict=False로 설정하여 불일치 허용

    # 마지막 레이어 수정 (전이 학습을 위해)
    model.fc1 = nn.Linear(model.flattened_size, 1024).to(device)  # fc1 수정
    model.fc2 = nn.Linear(1024, 254).to(device)  # fc2 수정
    model.fc3 = nn.Linear(254, num_classes).to(device)  # fc3 수정
    #################################################################
    
    # 옵티마이저 및 손실 함수 설정
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 학습 손실 기록
    train_losses = []

    # 모델을 학습 모드로 설정
    model.train()

    # 전이 학습 수행
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        # for images, labels in train_loader:
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')):
            images = images.to(device)
            labels = labels.to(device)

            # 순전파
            outputs = model(images)
                      
            loss = criterion(outputs, labels) / accumulation_steps

            # 역전파
            loss.backward()
            
            # accumulation_steps마다 가중치 업데이트
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item()
            
            # 메모리 정리 및 사용량 출력 (100 배치마다)
            if (i + 1) % 100 == 0:
                del outputs, loss   # 메모리 정리: Tensor 객체 삭제
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()    # 캐시된 메모리 정리

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

        # 에포크당 평균 손실 계산
        train_loss = running_loss / len(train_loader)

        # 에포크당 평균 손실 추가
        train_losses.append(train_loss)  

    # 모델 저장
    torch.save(model.state_dict(), 'fine_tuned_model.pth')
    print("모델 저장 완료!!")

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('에포크')
    plt.ylabel('손실')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # 그래프를 파일로 저장: plt.show() 보다 먼저 실행
    plt.savefig('./학습결과/transfer_results.png')  # 그래프를 'training_results.png'로 저장    
    plt.show()

if __name__ == '__main__':
    main()