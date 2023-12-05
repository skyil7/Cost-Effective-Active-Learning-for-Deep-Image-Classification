import torch
from torch import nn
import os

class CNN(nn.Module):
    def __init__(self, random_state=42):
        super(CNN, self).__init__()
        self.random_state = random_state

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
        )
        self.hidden_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout_1 = nn.Dropout(p=0.25)
        self.hidden_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.hidden_layer_3 = nn.Sequential(
            nn.Linear(in_features=64*8*8, out_features=512),
            nn.ReLU()
        )
        self.dropout_2 = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(in_features=512, out_features=10)
        self.initialize()

    def initialize(self):
        init_model = os.path.join("./model", f"init_seed_{self.random_state}.pth")
        if os.path.exists(init_model):
            self.load_state_dict(torch.load(init_model))
            return
        torch.save(self.state_dict(), init_model)


    def forward(self, input):
        """
        input shape: (3, 32, 32)
        output shape: (10)
        """
        x = self.input_layer(input)
        x = self.hidden_layer_1(x)
        x = self.maxpool(x)
        x = self.dropout_1(x)
        x = self.hidden_layer_2(x)
        x = self.maxpool(x)
        x = self.dropout_1(x)
        x = self.flatten(x)
        x = self.hidden_layer_3(x)
        x = self.dropout_2(x)
        x = self.classifier(x)
        return x
    
class CNN_IMU(nn.Module):
    def __init__(self, random_state=42):
        super(CNN_IMU, self).__init__()
        self.random_state = random_state

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
        )
        self.hidden_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout_1 = nn.Dropout(p=0.25)
        self.hidden_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.hidden_layer_3 = nn.Sequential(
            nn.Linear(in_features=64*56*56, out_features=512),
            nn.ReLU()
        )
        self.dropout_2 = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(in_features=512, out_features=6)
        self.initialize()

    def initialize(self):
        init_model = os.path.join("./model", f"imu_init_seed_{self.random_state}.pth")
        if os.path.exists(init_model):
            self.load_state_dict(torch.load(init_model))
            return
        torch.save(self.state_dict(), init_model)


    def forward(self, input):
        """
        input shape: (3, 32, 32)
        output shape: (10)
        """
        x = self.input_layer(input)
        x = self.hidden_layer_1(x)
        x = self.maxpool(x)
        x = self.dropout_1(x)
        x = self.hidden_layer_2(x)
        x = self.maxpool(x)
        x = self.dropout_1(x)
        x = self.flatten(x)
        try:
            x = self.hidden_layer_3(x)
            x = self.dropout_2(x)
            x = self.classifier(x)
        except:
            print(x.shape, "is not fit for model")
        return x
    
from torchvision import models

class CNN_CUB(nn.Module):
    def __init__(self, random_state=42):
        super(CNN_CUB, self).__init__()
        self.random_state = random_state

        # self.input_layer = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=1, padding=1),
        #     nn.ReLU(),
        # )
        # self.hidden_layer_1 = nn.Sequential(
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
        #     nn.ReLU(),
        # )
        # self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        # self.dropout_1 = nn.Dropout(p=0.25)
        # self.hidden_layer_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
        #     nn.ReLU(),
        # )
        # self.flatten = nn.Flatten()
        # self.hidden_layer_3 = nn.Sequential(
        #     nn.Linear(in_features=64*56*56, out_features=512),
        #     nn.ReLU()
        # )
        # self.dropout_2 = nn.Dropout(p=0.5)
        # self.classifier = nn.Linear(in_features=512, out_features=20)
        self.pretrained_resnet = models.resnet18(pretrained=True)  # 사전학습된 ResNet18 모델을 불러오기
        self.pretrained_resnet.fc = nn.Linear(512, 20)  # ResNet의 마지막 계층을 CIFAR10에 맞게 10의 출력 크기를 갖도록 수정 
        
        self.initialize()

    def initialize(self):
        init_model = os.path.join("./model", f"cub_init_seed_{self.random_state}.pth")
        if os.path.exists(init_model):
            self.load_state_dict(torch.load(init_model))
            return
        torch.save(self.state_dict(), init_model)

    def forward(self, input):
        return self.pretrained_resnet(input)

    # def forward(self, input):
    #     """
    #     input shape: (3, 32, 32)
    #     output shape: (10)
    #     """
    #     x = self.input_layer(input)
    #     x = self.hidden_layer_1(x)
    #     x = self.maxpool(x)
    #     x = self.dropout_1(x)
    #     x = self.hidden_layer_2(x)
    #     x = self.maxpool(x)
    #     x = self.dropout_1(x)
    #     x = self.flatten(x)
    #     try:
    #         x = self.hidden_layer_3(x)
    #         x = self.dropout_2(x)
    #         x = self.classifier(x)
    #     except:
    #         print(x.shape, "is not fit for model")
    #     return x