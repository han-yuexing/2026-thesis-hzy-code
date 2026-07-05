import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np

# 注意力图投影模块（卷积层+ReLU+GroupNorm）
class AttentionMapProjection(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=32):
        super(AttentionMapProjection, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.gn = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        # x = self.relu(self.gn(self.conv1(x)))
        # x = self.relu(self.gn(self.conv2(x)))
        # x = self.relu(self.gn(self.conv3(x)))
        # return x
        x1 = self.relu(self.gn(self.conv1(x)))
        x2 = self.relu(self.gn(self.conv2(x1)))
        x3 = self.relu(self.gn(self.conv3(x2)))
        return x3
        # return x3 + x  # 加入残差连接

# 通道重加权模块（两层全连接层+ReLU+Sigmoid）
# class ChannelReweighting(nn.Module):
#     def __init__(self, in_channels, hidden_channels):
#         super(ChannelReweighting, self).__init__()
#         self.fc1 = nn.Linear(in_channels, hidden_channels)
#         self.fc2 = nn.Linear(hidden_channels, in_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.sigmoid(self.fc2(x))
#         return x
class ChannelReweighting(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(ChannelReweighting, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)  # 第二个全连接层
        self.fc3 = nn.Linear(hidden_channels, in_channels)  # 第三个全连接层
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))  # 使用第二个全连接层
        x = self.sigmoid(self.fc3(x))  # 使用第三个全连接层
        
        return x


# 将上述模块组合为SAPM模块
class SAPM(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_groups=32):
        super(SAPM, self).__init__()
        self.attention_map_projection = AttentionMapProjection(in_channels, out_channels, num_groups)
        self.channel_reweighting = ChannelReweighting(out_channels, hidden_channels)

    def forward(self, x):
        # 注意力图投影
        attention_maps = self.attention_map_projection(x)  # 输出形状 (batch_size, out_channels, H, W)

        # 平均池化以获取通道描述符，不使用squeeze，避免丢失批量维度
        pooled_features = F.adaptive_avg_pool2d(attention_maps, (1, 1)).view(attention_maps.size(0), -1)  # (batch_size, out_channels)

        # 通道重加权 获取通道权重
        channel_weights = self.channel_reweighting(pooled_features)  # 输出形状 (batch_size, out_channels)

        # 调整 channel_weights 形状为 (batch_size, out_channels, 1, 1)
        channel_weights = channel_weights.view(channel_weights.size(0), channel_weights.size(1), 1, 1)

        # 将通道权重应用到注意力图上
        reweighted_features = attention_maps * channel_weights  # 元素乘法 (batch_size, out_channels, H, W)
        return reweighted_features