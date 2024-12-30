import torch.nn as nn
import math
import numpy as np
import torch


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)


class Model(nn.Module):

    def __init__(self, len_parts, num_classes, num_joints,
                 num_frames, num_heads, num_persons, num_channels,
                 kernel_size, use_pes=True, config=None,
                 att_drop=0, dropout=0, dropout2d=0, ter=False):
        super().__init__()

        self.len_parts = len_parts
        in_channels = config[0][0]
        self.out_channels = config[-1][1]
        self.input_map = nn.Sequential(
            nn.Conv2d(3, (in_channels // 2), 1),
            nn.BatchNorm2d((in_channels // 2)),
            nn.LeakyReLU(0.1),
        )
        self.diff_map1 = nn.Sequential(
            nn.Conv2d(3, in_channels // 8, 1),
            nn.BatchNorm2d(in_channels // 8),
            nn.LeakyReLU(0.1),
        )
        self.diff_map2 = nn.Sequential(
            nn.Conv2d(3, in_channels // 8, 1),
            nn.BatchNorm2d(in_channels // 8),
            nn.LeakyReLU(0.1),
        )
        self.diff_map3 = nn.Sequential(
            nn.Conv2d(3, in_channels // 8, 1),
            nn.BatchNorm2d(in_channels // 8),
            nn.LeakyReLU(0.1),
        )
        self.diff_map4 = nn.Sequential(
            nn.Conv2d(3, in_channels // 8, 1),
            nn.BatchNorm2d(in_channels // 8),
            nn.LeakyReLU(0.1),
        )
        self.l1 = All_TransFormer(64, 64, 16, num_frames=64, part=[1, 1, 1])
        self.l2 = All_TransFormer(64, 64, 16, num_frames=64, part=[1, 1, 1])
        self.l3 = All_TransFormer(64, 128, 32, num_frames=64, part=[1, 1, 1])
        self.l4 = All_TransFormer(128, 128, 32, num_frames=32, part=[1, 1, 1])
        self.l5 = All_TransFormer(128, 256, 64, num_frames=32, part=[1, 1, 1])
        self.l6 = All_TransFormer(256, 256, 64, num_frames=16, part=[1, 1, 1])
        self.l7 = All_TransFormer(256, 256, 64, num_frames=16, part=[1, 1, 1])
        self.l8 = All_TransFormer(256, 256, 64, num_frames=16, part=[1, 1, 1])
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0)
        self.fc = nn.Linear(self.out_channels, num_classes)
        self.drop_out = nn.Dropout(dropout)
        self.drop_out2d = nn.Dropout2d(dropout2d)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)

    def forward(self, x):

        N, C, T, V, M = x.shape

        x = x.view(N, C, T, V * M)  # 合并关节点

        dif1 = x[:, :, 1:] - x[:, :, 0:-1]
        dif1 = torch.cat([dif1.new(N, C, 1, V * M).zero_(), dif1], dim=-2)

        dif2 = x[:, :, 1:] - x[:, :, 0:-1]
        dif2 = torch.cat([dif2.new(N, C, 1, V * M).zero_(), dif2], dim=-2)

        dif3 = x[:, :, :-1] - x[:, :, 1:]
        dif3 = torch.cat([dif3, dif3.new(N, C, 1, V * M).zero_()], dim=-2)

        dif4 = x[:, :, :-1] - x[:, :, 1:]
        dif4 = torch.cat([dif4, dif4.new(N, C, 1, V * M).zero_()], dim=-2)

        x = torch.cat(
            (self.input_map(x), self.diff_map1(dif1), self.diff_map2(dif2), self.diff_map3(dif3), self.diff_map4(dif4)),
            dim=1)
        # res1 = self.res1(x)
        x = self.l1(x)
        # x = self.relu1(x + res1)
        # res2 = self.res2(x)
        x = self.l2(x)
        # x = self.relu2(x + res2)
        # res3 = self.res3(x)
        x = self.l3(x)
        x = self.maxpool(x)
        # x = self.relu3(x + res3)
        # res3 = self.res3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.maxpool(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        # NM, C, T, V
        x = x.view(N, M, self.out_channels, -1).contiguous()

        x = x.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1).contiguous()  # 1, 2, 256, 7500

        x = self.drop_out2d(x)
        x = x.mean(3).mean(1)

        x = self.drop_out(x)

        return self.fc(x)


class All_TransFormer(nn.Module):

    def __init__(self, in_channels, out_channels, qkv_dim, num_frames, part):
        super(All_TransFormer, self).__init__()
        self.ssa = STA_Block_att(in_channels, out_channels, qkv_dim, num_frames=num_frames, num_joints=50, num_heads=3, kernel_size=[1, 1])
        self.tta = TTA_Block_att(out_channels, out_channels, qkv_dim, num_frames=num_frames, num_joints=50, num_heads=3, kernel_size=[1, 1])

    def forward(self, x):
        x = self.ssa(x)
        x = self.tta(x)
        return x


class STA_Block_att(nn.Module):

    def __init__(self, in_channels, out_channels, qkv_dim,
                 num_frames, num_joints, num_heads,
                 kernel_size, use_pes=True, att_drop=0):
        super().__init__()
        self.qkv_dim = qkv_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_pes = use_pes
        pads = int((kernel_size[1] - 1) / 2)

        # Spatio-Temporal Tuples Attention
        if self.use_pes: self.pes = Pos_Embed(in_channels, num_frames, num_joints)
        self.to_qkvs = nn.Conv2d(in_channels, 2 * num_heads * qkv_dim, 1, bias=True)
        self.alphas = nn.Parameter(torch.ones(1, num_heads, 1, 1), requires_grad=True)
        self.att0s = nn.Parameter(torch.ones(1, num_heads, num_joints, num_joints) / num_joints, requires_grad=True)
        self.out_nets = nn.Sequential(
            nn.Conv2d(in_channels * num_heads, out_channels, (1, kernel_size[1]), padding=(0, pads)),
            nn.BatchNorm2d(out_channels))
        self.ff_net = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        if in_channels != out_channels:
            self.ress = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
            self.rest = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.ress = lambda x: x
            self.rest = lambda x: x

        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(att_drop)

    def forward(self, x):

        N, C, T, V = x.size()
        # Spatio-Temporal Tuples Attention
        xs = self.pes(x) + x if self.use_pes else x
        q, k = torch.chunk(self.to_qkvs(xs).view(N, 2 * self.num_heads, self.qkv_dim, T, V), 2, dim=1)
        attention = self.tan((torch.einsum('nhctu,nhctv->nhuv', [q, k]).contiguous() / (self.qkv_dim * T)).contiguous()) * self.alphas
        attention = attention + self.att0s.repeat(N, 1, 1, 1)
        attention = self.drop(attention)
        xs = torch.einsum('nctu,nhuv->nhctv', [x, attention]).contiguous().view(N, self.num_heads * self.in_channels, T,
                                                                                V)
        x_ress = self.ress(x)
        xs = self.relu(self.out_nets(xs) + x_ress)
        xs = self.relu(self.ff_net(xs) + x_ress)

        # Inter-Frame Feature Aggregation
        # xt = self.relu(self.out_nett(xs) + self.rest(xs))

        return xs


class TTA_Block_att(nn.Module):
    def __init__(self, in_channels, out_channels, qkv_dim,
                 num_frames, num_joints, num_heads,
                 kernel_size, use_pes=True, att_drop=0):
        super().__init__()
        self.qkv_dim = qkv_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_pes = use_pes
        pads = int((kernel_size[1] - 1) / 2)
        padt = int((kernel_size[0] - 1) / 2)

        if self.use_pes: self.pes = Ter_Embed(in_channels, num_frames, num_joints)
        self.to_qkvs = nn.Conv2d(in_channels, 2 * num_heads * qkv_dim, 1, bias=True)
        self.alphas = nn.Parameter(torch.ones(1, num_heads, 1, 1), requires_grad=True)

        self.att1s = nn.Parameter(torch.zeros(1, num_heads, num_frames, num_frames) + torch.eye(num_frames),
                                  requires_grad=True)
        self.out_nets = nn.Sequential(
            nn.Conv2d(in_channels * num_heads, out_channels, (1, kernel_size[1]), padding=(0, pads)),
            nn.BatchNorm2d(out_channels))
        self.ff_net = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))

        self.out_nett = nn.Sequential(nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), padding=(padt, 0)),
                                      nn.BatchNorm2d(out_channels))

        if in_channels != out_channels:
            self.ress = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
            self.rest = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.ress = lambda x: x
            self.rest = lambda x: x

        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(att_drop)

    def forward(self, x):  #
        N, C, T, V = x.size()

        xs = self.pes(x) + x if self.use_pes else x  # 位置编码

        q, k = torch.chunk(self.to_qkvs(xs).view(N, 2 * self.num_heads, self.qkv_dim, T, V), 2, dim=1)
        attention = (self.tan(torch.einsum('nhctv,nhcqv->nhtq', [q, k]) / (self.qkv_dim * V)).contiguous() * self.alphas).contiguous()
        attention = attention + self.att1s.repeat(N, 1, 1, 1)
        attention = self.drop(attention)
        xs = torch.einsum('nctv,nstq->nscqv', [x, attention]).contiguous().view(N, self.num_heads * self.in_channels, T, V)
        x_ress = self.ress(x)  # 升为out_channle通道
        xs = self.relu(self.out_nets(xs) + x_ress)  # in_channels * num_heads-> out_channels
        xs = self.relu(self.ff_net(xs) + x_ress)
        return xs


class Ter_Embed(nn.Module):
    def __init__(self, channels, num_frames, num_joints):
        super().__init__()

        pos_list = []
        for tk in range(num_frames):
            for st in range(num_joints):
                pos_list.append(tk)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()

        pe = torch.zeros(num_frames * num_joints, channels)

        div_term = torch.exp(torch.arange(0, channels, 2).float() * -(math.log(10000.0) / channels))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(num_frames, num_joints, channels).permute(2, 0, 1).unsqueeze(0).contiguous()
        self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        x = self.pe[:, :, :x.size(2)]
        return x


class Pos_Embed(nn.Module):
    def __init__(self, channels, num_frames, num_joints):
        super().__init__()

        pos_list = []
        for tk in range(num_frames):
            for st in range(num_joints):
                pos_list.append(st)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()

        pe = torch.zeros(num_frames * num_joints, channels)

        div_term = torch.exp(torch.arange(0, channels, 2).float() * -(math.log(10000.0) / channels))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(num_frames, num_joints, channels).permute(2, 0, 1).unsqueeze(0).contiguous()
        self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        x = self.pe[:, :, :x.size(2)]
        return x
