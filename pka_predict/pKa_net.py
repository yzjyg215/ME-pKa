import torch.nn as nn
import torch.nn.functional as F
import torch

# 定义3D卷积块
class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(Conv3DBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv3d(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

# 定义3D残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv3DBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = Conv3DBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm3d(out_channels)
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out

class PkaNetBN4F20S21ELU_ONLYRES(nn.Module):

    def __init__(self):
        super(PkaNetBN4F20S21ELU_ONLYRES, self).__init__()
        # 20 input data channel, 5 x 5 x 5 square convolution kernel
        self.conv1 = Conv3DBlock(20, 64, kernel_size=3, stride=1, padding=1)
        self.residual_block1 = ResidualBlock(64, 64, stride=1)
        self.residual_block2 = ResidualBlock(64, 128, stride=2)
        self.residual_block3 = ResidualBlock(128, 256, stride=2)
        self.residual_block4 = ResidualBlock(256, 512, stride=2)
        self.residual_block5 = ResidualBlock(512, 512, stride=1)
        self.residual_block6 = ResidualBlock(512, 512, stride=1)
        self.residual_block7 = ResidualBlock(512, 512, stride=1)
        self.residual_block8 = ResidualBlock(512, 512, stride=1)
        self.residual_block9 = ResidualBlock(512, 1024, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(1024, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.residual_block1(out)
        out = self.residual_block2(out)
        out = self.residual_block3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.residual_block7(out)
        out = self.residual_block8(out)
        out = self.residual_block9(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class PkaNetBN4F20S21ELU(nn.Module):

    def __init__(self, fasta_num, fasta_layer):
        super(PkaNetBN4F20S21ELU, self).__init__()
        # 20 input data channel, 5 x 5 x 5 square convolution kernel
        self.conv1 = Conv3DBlock(20, 64, kernel_size=3, stride=1, padding=1)
        self.residual_block1 = ResidualBlock(64, 64, stride=1)
        self.residual_block2 = ResidualBlock(64, 128, stride=2)
        self.residual_block3 = ResidualBlock(128, 256, stride=2)
        self.residual_block4 = ResidualBlock(256, 512, stride=2)
        self.residual_block5 = ResidualBlock(512, 512, stride=1)
        self.residual_block6 = ResidualBlock(512, 512, stride=1)
        self.residual_block7 = ResidualBlock(512, 512, stride=1)
        self.residual_block8 = ResidualBlock(512, 512, stride=1)
        self.residual_block9 = ResidualBlock(512, 1024, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fasta_bn = nn.LayerNorm(fasta_num * fasta_layer)
        self.fc1 = nn.Linear(1024 + fasta_num * fasta_layer, 1024)
        self.relu1 = nn.ReLU()
        self.droupt1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024,256)
        self.relu2 = nn.ReLU()
        self.droupt2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256,fasta_num)

    def forward(self, x, fasta):
        out = self.conv1(x)
        out = self.residual_block1(out)
        out = self.residual_block2(out)
        out = self.residual_block3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.residual_block7(out)
        out = self.residual_block8(out)
        out = self.residual_block9(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        #print(out.shape,fasta.shape)
        out = torch.concat((out, self.fasta_bn(fasta)), dim=1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.droupt1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.droupt2(out)
        out = self.fc3(out)
        return out
