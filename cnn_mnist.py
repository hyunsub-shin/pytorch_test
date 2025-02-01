import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt
import platform  # 폰트관련 운영체제 확인
import os
from PIL import Image
import torch.nn.functional as F
from models.convnet import ConvNet
import torch.multiprocessing as mp
from torch.multiprocessing import freeze_support
import psutil # CPU 메모리 사용량 출력
from tqdm import tqdm

##############################################################################################
# pytorch cuda version 12.1 설치 기준
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
##############################################################################################

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
    
    ################################################
    # 하이퍼파라미터 설정
    num_epochs = 30     # 에포크 수 조정
    batch_size = 32     # 배치 사이즈 조정 
    
    # 학습률 조정 0 ~ 1 사이의 작은 값 사용(예: 0.1, 0.01, 0.001, 0.0001, ...)
    # 큰 학습률: 빠르게 학습, 작은 학습률: 정확도 향상
    learning_rate = 0.0005  

    accumulation_steps = 4  # 그래디언트 누적 스텝 수

    # ConvNet 입력 이미지 크기 설정
    input_height = 28   # 입력 이미지 높이
    input_width = 28    # 입력 이미지 너비
    ################################################
    
    # 시스템 정보는 시작할 때 한 번만 출력
    print("\n=== 시스템 정보 ===")
    print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")
    print(f"현재 PyTorch의 CUDA 버전: {torch.version.cuda}")
    print(f"PyTorch 버전: {torch.__version__}") # 예: '2.0.0+cu121'는 CUDA 12.1 버전을 지원
    print(f'사용 중인 장치: {device}')
    print('-' * 50)
    
    if device.type == 'cuda':        
        print(f'현재 사용 중인 GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU 메모리 사용량:')
        print(f'할당된 메모리: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB')
        print(f'캐시된 메모리: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB')
        
        # CUDA 설정
        torch.backends.cudnn.benchmark = True # 속도 향상을 위한 설정
        torch.backends.cudnn.deterministic = True # 재현 가능성 확보
        torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat-32 사용
        
        torch.cuda.empty_cache()
        # 메모리 할당 모드 설정
        torch.cuda.set_per_process_memory_fraction(0.8)  # GPU 메모리의 80% 사용
    else:
        memory_info = psutil.virtual_memory()   # 전체 메모리 정보 가져오기
        
        # 전체 메모리, 사용 중인 메모리, 사용 가능한 메모리 출력
        print(f'전체 메모리: {memory_info.total / (1024 ** 2):.2f} MB')
        print(f'사용 중인 메모리: {memory_info.used / (1024 ** 2):.2f} MB')
        print(f'메모리 사용률: {memory_info.percent}%')
        
    # 데이터 전처리 설정
    transform_train = transforms.Compose([
        transforms.Resize((input_height, input_width)),
        transforms.RandomHorizontalFlip(),  # 좌우 반전
        transforms.RandomRotation(10),      # 회전
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 밝기, 대비, 채도, 색조 조정
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 평균/표준편차 사용
                            std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_height, input_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 평균/표준편차 사용
                            std=[0.229, 0.224, 0.225])
    ])

    # 커스텀 데이터셋 로드
    train_dataset = CustomDataset(
        data_dir='./data',
        dataset_type='train',
        transform=transform_train
    )

    test_dataset = CustomDataset(
        data_dir='./data',
        dataset_type='test',
        transform=transform_test  # 증강 없이 기본 전처리만 적용
    )

    # train_loader DataLoader 매개변수 미리 설정
    train_dataloader_params = {
        'dataset': train_dataset,
        'batch_size': batch_size,
        'shuffle': True
    }

    # GPU 사용 가능 여부에 따라 추가 매개변수 설정
    if device.type == 'cuda':
        train_dataloader_params.update({
            'num_workers': 4,      # CPU 워커 수 증가
            'pin_memory': True,    # CUDA 핀 메모리 사용
            'prefetch_factor': 2   # 데이터 프리페치
        })
    else:
        train_dataloader_params.update({
            'num_workers': 1,      # CPU 워커 수
            'pin_memory': False,   # CUDA 핀 메모리 비활성화
            'prefetch_factor': 1   # 데이터 프리페치
        })

    # # DataLoader 초기화
    # train_loader = torch.utils.data.DataLoader(
    #     dataset=train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True
    # )
    train_loader = torch.utils.data.DataLoader(**train_dataloader_params)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # 모델 초기화 전에 클래스 수 확인 및 출력
    num_classes = len(train_dataset.classes)  # classes 리스트의 길이로 계산
    print(f'감지된 클래스 수: {num_classes}')
    print(f'클래스 목록: {train_dataset.classes}')
    # 라벨 범위 확인
    print(f'라벨 범위: {min(train_dataset.labels)} ~ {max(train_dataset.labels)}')

    # 모델 초기화
    model = ConvNet(num_classes, input_height, input_width).to(device)

    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # 옵티마이저 변경 테스트(Nesterov, Adadelta, RMSprop, FTRL)

    # 학습률을 동적으로 조정
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1, # 학습률을 1/10로 감소
        patience=2 # 2번의 에포크 동안 성능 향상이 없으면 학습률 감소
    )
    
    ###############################
    # 조기 종료를 위한 변수들 초기화
    ###############################
    best_val_loss = float('inf')
    patience = 3  # 몇 번의 에포크동안 개선이 없으면 종료할지
    patience_counter = 0
    
    # 학습/검증 손실 기록
    train_losses = []
    val_losses = []
    
    # 학습/검증 정확도 기록
    train_accuracies = []
    val_accuracies = []
    
    print("\n=== 학습 시작 ===")
    
    # 학습 루프
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # for i, (images, labels) in enumerate(train_loader):
        # # tqdm으로 학습 진행 상황 표시
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')):
            images = images.to(device)
            labels = labels.to(device)
            
            # 순전파
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps
            
            # 역전파
            loss.backward()
            
            # 그래디언트 클리핑 추가
            '''
            작은 값 (예: 0.1, 0.5)
            - 더 보수적인 학습
            - 매우 안정적이지만 학습 속도가 느릴 수 있음
            큰 값 (예: 1.0, 2.0)
            - 더 적극적인 학습
            - 학습 속도는 빠르지만 불안정할 수 있음
            '''
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # accumulation_steps마다 가중치 업데이트
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # 통계
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # # for debug
            # # 100 배치마다 진행상황 출력
            # if (i + 1) % 100 == 0: 
            #     print(f'배치 [{i+1}/{len(train_loader)}], '
            #         f'배치 크기: {labels.size(0)}, '
            #         f'현재 손실: {loss.item():.4f}, '
            #         f'현재 정확도: {100 * (predicted == labels).sum().item() / labels.size(0):.2f}%')
                
            #     # 현재까지의 누적 통계도 출력
            #     print(f'누적 통계 - 전체 샘플: {total}, '
            #         f'정답 개수: {correct}, '
            #         f'평균 손실: {running_loss/(i+1):.4f}, '
            #         f'평균 정확도: {100 * correct/total:.2f}%')
            #     print('-' * 80)  # 구분선 출력
            
            # 메모리 정리 및 사용량 출력 (100 배치마다)
            if (i + 1) % 100 == 0:
                del outputs, loss   # 메모리 정리: Tensor 객체 삭제
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()    # 캐시된 메모리 정리
                    
                #     # GPU 메모리 사용량 출력
                #     print(f'배치 {i+1}/{len(train_loader)} - GPU 메모리:')
                #     print(f'할당된 메모리: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB')
                #     print(f'캐시된 메모리: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB')
                #     print('-' * 50)
                # else:
                #     memory_info = psutil.virtual_memory()   # 전체 메모리 정보 가져오기
                    
                #     # 전체 메모리, 사용 중인 메모리, 사용 가능한 메모리 출력
                #     print(f'전체 메모리: {memory_info.total / (1024 ** 2):.2f} MB')
                #     print(f'사용 중인 메모리: {memory_info.used / (1024 ** 2):.2f} MB')
                #     print(f'메모리 사용률: {memory_info.percent}%')
                #     print('-' * 50)
        
        # 에포크당 평균 손실과 정확도 계산
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # 검증 단계 추가
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(test_loader)
        val_acc = 100 * val_correct / val_total
        
        # 검증 손실에 따라 학습률 조정
        scheduler.step(val_loss)

        # # 현재 학습률 출력 for debug
        # current_lr = scheduler.get_last_lr()[0]  # 현재 학습률 가져오기
        # print(f'현재 학습률: {current_lr:.6f}')

        ###############################
        # 검증 손실에 따른 조기 종료 검사
        ###############################
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 최상의 모델 저장
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'조기 종료: {epoch+1} 에포크에서 학습 중단')
                break
        
        # ##############################
        # # for Debug 학습 진행중 표시
        # ##############################
        # print(f'에포크 [{epoch+1}/{num_epochs}]: '
        #       f'훈련 손실={train_loss:.4f}, 훈련 정확도={train_acc:.2f}%, '
        #       f'검증 손실={val_loss:.4f}, 검증 정확도={val_acc:.2f}%')
        # print(f'Gap: {abs(train_acc - val_acc):.2f}%')
        # print('=' * 50)

        train_losses.append(train_loss)  # 에포크당 평균 손실 추가
        val_losses.append(val_loss)      # 에포크당 평균 검증 손실 추가
        
        train_accuracies.append(train_acc)     # 에포크당 평균 정확도 추가 
        val_accuracies.append(val_acc)         # 에포크당 평균 검증 정확도 추가

    ############################
    # for debug 학습 완료후 표시
    ############################
    print(f'에포크 [{epoch+1}/{num_epochs}]: '
            f'훈련 손실={train_loss:.4f}, 훈련 정확도={train_acc:.2f}%, '
            f'검증 손실={val_loss:.4f}, 검증 정확도={val_acc:.2f}%')
    print(f'Gap: {abs(train_acc - val_acc):.2f}%')
    print('=' * 80)

    print('학습 완료!')

    # 학습 완료 후 모델 저장
    torch.save(model.state_dict(), 'trained_model.pth')
    print('모델이 저장되었습니다.')

    # ##################################################################
    # # 모델 가중치 출력 
    # weights = model.state_dict()
    # # 레이어 1, 2, 3, FC 레이어의 가중치 출력
    # print("Layer 1 : ", weights['layer1.0.weight'])
    # print("Layer 2 : ", weights['layer2.0.weight'])
    # print("Layer 3 : ", weights['layer3.0.weight'])
    # print("FC 1 : ", weights['fc1.weight'])
    # print("FC 2 : ", weights['fc2.weight'])

    # # 첫 번째 Conv2d 레이어의 가중치 시각화
    # weights_layer1 = weights['layer1.0.weight'].cpu().detach().numpy()
    # # 첫 번째 필터의 가중치 시각화
    # plt.imshow(weights_layer1[0, 0, :, :], cmap='gray')  # 첫 번째 필터의 첫 번째 채널
    # plt.colorbar()
    # plt.title('First Filter of First Conv Layer')
    # plt.show()
    # ##################################################################
    
    # 학습 과정 시각화
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('학습 손실 변화')
    plt.xlabel('에포크')
    plt.ylabel('손실')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('학습 정확도 변화')
    plt.xlabel('에포크')
    plt.ylabel('정확도 (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    # 그래프를 파일로 저장: plt.show() 보다 먼저 실행
    plt.savefig('./학습결과/training_results.png')  # 그래프를 'training_results.png'로 저장
    
    plt.show()

if __name__ == '__main__':
    freeze_support()  # Windows에서 필요
    main()

