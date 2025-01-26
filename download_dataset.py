import torchvision
import torchvision.transforms as transforms
import os
import shutil
from tqdm import tqdm
import requests
from PIL import Image
import numpy as np

def download_and_prepare_dataset(dataset_name='cifar10', data_dir='./data'):
    """
    데이터셋을 다운로드하고 train/test 폴더 구조로 준비
    """
    # 기본 transform 설정
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    
    # 데이터 디렉토리 생성
    os.makedirs(data_dir, exist_ok=True)
    
    if dataset_name.lower() == 'cifar10':
        # CIFAR-10 데이터셋 다운로드
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
        
        # 클래스 이름
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']
    
    elif dataset_name.lower() == 'mnist':
        # MNIST 데이터셋 다운로드
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        test_dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
        
        # 클래스 이름
        classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    # 데이터를 이미지 파일로 저장
    def save_dataset(dataset, is_train=True):
        split = 'train' if is_train else 'test'
        
        # 각 클래스별 디렉토리 생성
        for class_idx in range(len(classes)):
            class_dir = os.path.join(data_dir, split, classes[class_idx])
            os.makedirs(class_dir, exist_ok=True)
        
        # 데이터셋의 각 이미지를 저장
        for idx in tqdm(range(len(dataset)), desc=f'Saving {split} dataset'):
            img, label = dataset[idx]
            
            # 텐서를 PIL 이미지로 변환
            if dataset_name.lower() == 'mnist':
                img = transforms.ToPILImage()(img).convert('RGB')
            else:
                img = transforms.ToPILImage()(img)
            
            # 이미지 저장
            img_path = os.path.join(
                data_dir,
                split,
                classes[label],
                f'{classes[label]}_{idx}.png'
            )
            img.save(img_path)
    
    print("데이터셋 저장 시작...")
    
    # train 데이터 저장
    save_dataset(train_dataset, is_train=True)
    
    # test 데이터 저장
    save_dataset(test_dataset, is_train=False)
    
    print(f"\n데이터셋 준비 완료!")
    print(f"총 훈련 이미지 수: {len(train_dataset)}")
    print(f"총 테스트 이미지 수: {len(test_dataset)}")
    print(f"클래스: {classes}")

def download_custom_dataset(urls, labels, data_dir='./data'):
    """
    사용자 정의 URL에서 이미지를 다운로드
    """
    os.makedirs(data_dir, exist_ok=True)
    
    for label in labels:
        os.makedirs(os.path.join(data_dir, 'train', label), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'test', label), exist_ok=True)
    
    for url, label in tqdm(zip(urls, labels), desc='Downloading images'):
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                img = Image.open(response.raw)
                img = img.convert('RGB')
                
                # 파일명 생성
                filename = f"{label}_{url.split('/')[-1]}"
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filename += '.jpg'
                
                # 80%는 학습 데이터로, 20%는 테스트 데이터로
                if np.random.random() < 0.8:
                    save_path = os.path.join(data_dir, 'train', label, filename)
                else:
                    save_path = os.path.join(data_dir, 'test', label, filename)
                
                img.save(save_path)
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")

if __name__ == "__main__":
    # CIFAR-10 데이터셋 다운로드
    download_and_prepare_dataset('cifar10', './data')
    
    # 또는 MNIST 데이터셋 다운로드
    # download_and_prepare_dataset('mnist', './data')
    
    # 또는 사용자 정의 데이터셋 다운로드
    # custom_urls = [
    #     'https://example.com/cat1.jpg',
    #     'https://example.com/cat2.jpg',
    #     'https://example.com/dog1.jpg',
    #     'https://example.com/dog2.jpg'
    # ]
    # custom_labels = ['cat', 'cat', 'dog', 'dog']
    # download_custom_dataset(custom_urls, custom_labels, './data') 