import torch
import torch.nn as nn
import math
import torch.nn.functional as F
# ---------------------------- ResNet ---------------------------------
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation
    )

class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ECoGResNet(nn.Module):
    def __init__(self, num_channels=64, num_classes=5, layers=[2, 2, 2, 2], 
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # First conv layer: [B, C, T] -> [B, 64, T//2]
        self.conv1 = nn.Conv1d(num_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock1D, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock1D, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(BasicBlock1D, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(BasicBlock1D, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * BasicBlock1D.expansion, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock1D):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C, T] —— 注意：你的 Dataset 返回的就是 [C, T]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # [B, 512, 1]
        x = torch.flatten(x, 1)  # [B, 512]
        x = self.fc(x)
        return x
    


# ---------------------------- EEGNet ---------------------------------
class ECoGEEGNet(nn.Module):
    def __init__(self, num_channels, num_classes, input_time_length, 
                 dropoutRate=0.5, F1=8, D=2, F2=16, kernLength=64):
        super(ECoGEEGNet, self).__init__()
        
        self.num_channels = num_channels
        self.input_time_length = input_time_length
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, D * F1, (num_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(D * F1, D * F1, (1, 16), groups=D * F1, bias=False, padding=(0, 7)),
            nn.Conv2d(D * F1, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )
        
        # 动态计算展平后的维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, num_channels, input_time_length)
            x = self.block1(dummy_input)
            x = self.block2(x)
            n_out = x.view(1, -1).size(1)
        
        self.classifier = nn.Linear(n_out, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, C, T] -> [B, 1, C, T]
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ------------------------------ Transformer ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, L, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x


class ECoGTransformerClassifier(nn.Module):
    def __init__(
        self,
        num_channels=64,
        seq_len=256,
        embed_dim=128,
        nhead=8,
        num_layers=4,
        num_classes=5,
        dropout=0.1
    ):
        super().__init__()
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # Step 1: 将 [C, T] -> [T, embed_dim] via linear projection per time step
        # We treat each time point as a token, and project all channels into a vector
        self.input_proj = nn.Linear(num_channels, embed_dim)

        # Step 2: Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_len+1)

        # Step 3: Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Step 4: Classification head (use [CLS] or mean pooling)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: [B, C, T] → we want [B, T, C]
        x = x.permute(0, 2, 1)  # [B, T, C]

        # Project to embedding space
        x = self.input_proj(x)  # [B, T, embed_dim]

        # Add [CLS] token at the beginning
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, T+1, embed_dim]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer
        x = self.transformer(x)  # [B, T+1, embed_dim]

        # Use [CLS] token for classification
        cls_out = x[:, 0]  # [B, embed_dim]
        cls_out = self.dropout(cls_out)
        logits = self.classifier(cls_out)  # [B, num_classes]

        return logits
  