import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
from models.convnet import ConvNet  # ConvNet 클래스 임포트
import matplotlib.pyplot as plt
from PIL import Image
import platform # 폰트관련 운영체제 확인

def predict_image(image_path, model_path, data_dir):
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'사용 장치: {device}')
    
    # 클래스 목록 가져오기 (train 폴더 기준)
    train_dir = os.path.join(data_dir, 'train')
    classes = sorted(os.listdir(train_dir))
    num_classes = len(classes)
    print(f'감지된 클래스: {classes}')
      
    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 모델 초기화
    input_height = 28  # 입력 이미지 높이 # transform에서 적용한 사이즈
    input_width = 28   # 입력 이미지 너비 # transform에서 적용한 사이즈
    model = ConvNet(device, num_classes, input_height, input_width).to(device)
    
    # 모델 가중치 로드
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # 이미지 로드 및 변환
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # 예측
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        
    predicted_class = classes[predicted.item()]
    probabilities = outputs.softmax(1)[0]
    
    return predicted_class, probabilities, classes

def image_show(image_path, predicted_class):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')  # 축 숨기기
    plt.title(f'입력이미지: {image_path} \n예측된 클래스: [{predicted_class}]')  # 예측된 클래스 제목 추가
    plt.show()  # 이미지 출력

if __name__ == "__main__":
    # 한글 폰트 설정
    if platform.system() == 'Windows':
        font_name = 'Malgun Gothic'
    elif platform.system() == 'Darwin':
        font_name = 'AppleGothic'
    else:
        font_name = 'NanumGothic'

    plt.rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False
    
    # 예측 실행
    image_path = "./data/dog1.jpg"  # 예측하고 싶은 이미지 경로
    # image_path = "./data/cat1.jpg"  # 예측하고 싶은 이미지 경로
    model_path = "trained_model.pth"         # 저장된 모델 경로
    # model_path = "best_model.pth"         # 저장된 모델 경로
    data_dir = "./data"                      # 데이터 디렉토리 경로
    
    predicted_class, probabilities, classes = predict_image(image_path, model_path, data_dir)
    
    # 이미지 출력
    image_show(image_path, predicted_class)

    print('클래스별 확률:')
    for cls, prob in zip(classes, probabilities):
        print(f'{cls}: {prob.item():.4f}')
        
    print(f'예측된 클래스: {predicted_class}')