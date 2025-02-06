import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, num_classes, input_height, input_width):
        super(ConvNet, self).__init__()
        
        # CNN 모델 정의
        self.layer1 = nn.Sequential(
            # (ex): [3,28,28] -> [32,28,28]
            # out = (input - kernel + 2*padding)/stride + 1 : 28 -> 28
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False), #흑백 이미지의 경우 1, 컬러 이미지의 경우 3
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # [32,28,28] -> [64,28,28]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # [64,28,28] -> [64,14,14]
            nn.MaxPool2d(2)     # 첫 번째 pooling layer
            # 첫 번째 층에는 Dropout 적용하지 않는 것이 일반적인 권장 사항
            # nn.Dropout2d(0.3)  # CNN용 Dropout
        )
        self.layer2 = nn.Sequential(
            # [64,14,14] -> [128,14,14]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # [128,14,14] -> [256,14,14]
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # [256,14,14] -> [256,7,7]
            nn.MaxPool2d(2),    # 두 번째 pooling layer
            nn.Dropout2d(0.3)  # CNN용 Dropout
        )
        self.layer3 = nn.Sequential(
            # [256,7,7] -> [512,7,7]
            nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # [512,7,7] -> [1024,7,7]
            nn.Conv2d(512, 1024, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            # [1024,7,7] -> [1024,3,3]
            nn.MaxPool2d(2),    # 세 번째 pooling layer
            nn.Dropout2d(0.3)  # CNN용 Dropout
        )
        
        # # 입력 이미지가 28x28일 때의 계산
        # # 28x28 -> 14x14 -> 7x7 -> 3x3
        # # 따라서 마지막 특성 맵의 크기는 1024 * 3 * 3
        # self.fc1 = nn.Linear(1024 * 3 * 3, 2048)
        # self.fc2 = nn.Linear(2048, num_classes)
        # self.dropout = nn.Dropout(0.5)
        
        ###########################
        # Flatten된 입력 크기 계산
        ###########################
        '''
        입력 이미지 크기가 input_height와 input_width일 때:
        layer1 후:
            높이: input_height // 2
            너비: input_width // 2
        layer2 후:
            높이: input_height // 4
            너비: input_width // 4
        layer3 후:
            높이: input_height // 8
            너비: input_width // 8
        '''
        self.out_ch_num = 1024 # 마지막 layer의 채널 수
        self.flattened_size = self.out_ch_num * (input_height // 8) * (input_width // 8)
        
        # FC 레이어 설정
        self.fc1 = nn.Linear(self.flattened_size, 512)  
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # FC 레이어용 Dropout
        self.dropout = nn.Dropout(0.5) # FC 레이어는 더 높은 dropout 비율 사용
        
    def forward(self, x):
        # CNN 레이어 통과
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # Flatten
        # out.size(0) => batch_size를 나타냄
        out = out.view(out.size(0), -1)
        
        # FC 레이어 통과
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout(out) # FC 레이어 후에 dropout 적용
        out = self.fc2(out)     
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)     #마지막 레이어에는 dropout 적용하지 않음
        
        return out

# Fully Convolutional Network
class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1x1 Convolutional Layers (FC Layer 대체)
        self.conv3 = nn.Conv2d(128, num_classes, kernel_size=1)

        # # Upsampling Layer (Deconvolution)
        # self.upsample = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # GAP layer 추가

    def forward(self, x):
        # Convolutional Layers
        out = self.pool1(self.relu1(self.conv1(x)))
        out = self.pool2(self.relu2(self.conv2(out)))

        # 1x1 Convolutional Layer
        out = self.conv3(out)

        # # Upsampling
        # out = self.upsample(out)
        
        # # FCN 출력 크기 조정 (interpolation 사용)
        # out = F.interpolate(out, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        
        # # Adaptive Average Pooling 추가
        # # 입력 텐서의 크기에 관계없이 출력 텐서의 크기를 (batch_size, num_channels, 1, 1)로 만듦
        out = self.global_avg_pool(out) # GAP layer 통과 
        
        # criterion = nn.CrossEntropyLoss() 의 tensor size에 맞게 변경        
        out = torch.flatten(out, 1)  # Flatten     

        return out

# transfer FCN(if pretrained_cnn) or ConvNet + FCN(if not pretrained_cnn)
class FCNN(nn.Module):
    def __init__(self, num_classes, input_height, input_width, convnet_out_ch_num, pretrained_cnn=None):
        super(FCNN, self).__init__()

        if pretrained_cnn:
            # retrained model이 있을 경우 pretrained model layer를 가져옴
            self.features = nn.Sequential(*list(pretrained_cnn.children())[:-3])    # 마지막 fc layer 3개 제거
        else:
            # ConvNet 인스턴스 생성
            temp_cnn = ConvNet(num_classes, input_height, input_width)

            # ConvNet의 convolutional layer 부분 복사 (layer1, layer2, layer3)
            self.features = nn.Sequential(
                temp_cnn.layer1,
                temp_cnn.layer2,
                temp_cnn.layer3
            )

        # 1x1 Convolution Layer (FC Layer 대체) - 채널 수 조절
        self.conv4 = nn.Conv2d(convnet_out_ch_num, 512, kernel_size=1)  # 채널 수 1024 -> 512
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout4 = nn.Dropout2d(0.3)

        self.conv5 = nn.Conv2d(512, 256, kernel_size=1) # 채널 수 512 -> 256
        self.relu5 = nn.ReLU(inplace=True)
        self.dropout5 = nn.Dropout2d(0.3)

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1) # 마지막 채널은 num_classes

        # # Upsampling Layer (Deconvolution) - 필요에 따라 조절
        # self.upsample = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4) # kernel size, stride, padding 조절

        # Adaptive Average Pooling 추가
        # 입력 텐서의 크기에 관계없이 출력 텐서의 크기를 (batch_size, num_channels, 1, 1)로 만듦
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 입력 이미지 크기 저장
        self.input_height = input_height
        self.input_width = input_width

    def forward(self, x):
        # ConvNet의 convolutional layer 통과
        out = self.features(x) # self.layer1, self.layer2, self.layer3(x) 대체

        # 1x1 Convolutional Layer
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.dropout4(out)

        out = self.conv5(out)
        out = self.relu5(out)
        out = self.dropout5(out)

        # Classification (1x1 conv)
        out = self.classifier(out)

        # # Upsampling
        # out = self.upsample(out)
        
        # Adaptive Average Pooling 적용
        out = self.avgpool(out)
        
        # criterion = nn.CrossEntropyLoss() 의 tensor size에 맞게 변경
        out = torch.flatten(out, 1)  # Flatten

        return out

def create_fcn_transfer_model(num_classes, input_height, input_width, pretrained_model_path=None):
    cnn_model = ConvNet(num_classes, input_height, input_width) # ConvNet 인스턴스 생성
    convnet_out_ch_num = cnn_model.out_ch_num # ConvNet의 out_ch_num 가져오기
    
    # pre-training된 weight가 있다면 load
    if pretrained_model_path:
        cnn_model.load_state_dict(torch.load(pretrained_model_path, weights_only=True))

        # FCN 모델 생성 (ConvNet의 convolutional layer 부분 복사)
        fcn_model = FCNN(num_classes, input_height, input_width, convnet_out_ch_num, pretrained_cnn=cnn_model)
    else:
        fcn_model = FCNN(num_classes, input_height, input_width, convnet_out_ch_num)

    return fcn_model

# # 사용 예시
# num_classes = 10  # 실제 클래스 개수로 변경
# input_height = 224 # input image의 height
# input_width = 224 # input image의 width

# # 1. pretrained model 없이 FCN 생성
# fcn_model = create_fcn_transfer_model(num_classes, input_height, input_width)

# # 2. pretrained model 기반 FCN 생성
# pretrained_model_path = 'pretrained_model.pth'  # pre-training된 모델 경로
# fcn_model_transfer = create_fcn_transfer_model(num_classes, input_height, input_width, pretrained_model_path=pretrained_model_path)

# # 모델 사용
# # input_image = torch.randn(1, 3, input_height, input_width)  # 예시 입력 이미지
# # output = fcn_model(input_image)  # 모델 실행
# # print(output.shape)  # 출력 크기 확인