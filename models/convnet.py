import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, device, num_classes, input_height, input_width):
        super(ConvNet, self).__init__()
        
        # CNN 모델 정의
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), #흑백 이미지의 경우 1, 컬러 이미지의 경우 3
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)     # 첫 번째 pooling layer
            # 첫 번째 층에는 Dropout 적용하지 않는 것이 일반적인 권장 사항
            # nn.Dropout2d(0.3)  # CNN용 Dropout
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),    # 두 번째 pooling layer
            nn.Dropout2d(0.3)  # CNN용 Dropout
        )
        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 1024, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),    # 세 번째 pooling layer
        #     nn.Dropout2d(0.5)  # CNN용 Dropout
        # )
        
        # # 입력 이미지가 28x28일 때의 계산
        # # 28x28 -> 14x14 -> 7x7 -> 3x3
        # # 따라서 마지막 특성 맵의 크기는 128 * 3 * 3
        # self.fc1 = nn.Linear(128 * 3 * 3, 2048)
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
        self.out_ch_num = 256 # 마지막 layer의 채널 수
        self.flattened_size = self.out_ch_num * (input_height // 4) * (input_width // 4)
        
        # GPU/CPU에 따른 FC 레이어 설정
        if device.type == 'cuda':
            self.fc1 = nn.Linear(self.flattened_size, 1024)
            self.fc2 = nn.Linear(1024, num_classes)          
        else:
            self.fc1 = nn.Linear(self.flattened_size, 512)       
            self.fc2 = nn.Linear(512, num_classes)

        # FC 레이어용 Dropout
        self.dropout = nn.Dropout(0.5) # FC 레이어는 더 높은 dropout 비율 사용
        
    def forward(self, x):
        # CNN 레이어 통과
        out = self.layer1(x)
        out = self.layer2(out)
        # out = self.layer3(out)
        
        # Flatten
        out = out.view(out.size(0), -1)
        
        # FC 레이어 통과
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout(out) # FC 레이어 후에 dropout 적용
        out = self.fc2(out)  #마지막 레이어에는 dropout 적용하지 않음
        
        return out