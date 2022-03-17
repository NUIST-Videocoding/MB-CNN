import torch
import torch.nn as nn
import torch.nn.functional as F
import common


class ResBlock_128_1(nn.Module):
    def __init__(self):
        super(ResBlock_128_1, self).__init__()
        self.ResBlock_128_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
        )
        self.conv = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2)

    def forward(self, x):
        x_ = self.conv(x)
        return x_ + self.ResBlock_128_1(x)


class ResBlock_128_2(nn.Module):
    def __init__(self):
        super(ResBlock_128_2, self).__init__()
        self.ResBlock_128_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
        )

    def forward(self, x):
        return x + self.ResBlock_128_2(x)


class ResBlock_256_1(nn.Module):
    def __init__(self):
        super(ResBlock_256_1, self).__init__()
        self.ResBlock_256_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
        )
        self.conv = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2)

    def forward(self, x):
        x_ = self.conv(x)
        return x_ + self.ResBlock_256_1(x)


class ResBlock_256_2(nn.Module):
    def __init__(self):
        super(ResBlock_256_2, self).__init__()
        self.ResBlock_256_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
        )

    def forward(self, x):
        return x + self.ResBlock_256_2(x)


class ResBlock_512_1(nn.Module):
    def __init__(self):
        super(ResBlock_512_1, self).__init__()
        self.ResBlock_512_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
        )

        self.conv = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2)

    def forward(self, x):
        x_ = self.conv(x)
        return x_ + self.ResBlock_512_1(x)


class ResBlock_512_2(nn.Module):
    def __init__(self):
        super(ResBlock_512_2, self).__init__()
        self.ResBlock_512_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
        )

    def forward(self, x):
        return x + self.ResBlock_512_2(x)


class CNNIQAnet2(nn.Module):
    def __init__(self):
        super(CNNIQAnet2, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)

        ResBlock1_1 = []
        ResBlock1_1.append(ResBlock_128_1())
        self.ResBlock1_1 = nn.Sequential(*ResBlock1_1)

        ResBlock1_2 = []
        ResBlock1_2.append(ResBlock_128_2())
        self.ResBlock1_2 = nn.Sequential(*ResBlock1_2)

        ResBlock2_1 = []
        ResBlock2_1.append(ResBlock_256_1())
        self.ResBlock2_1 = nn.Sequential(*ResBlock2_1)

        ResBlock2_2 = []
        ResBlock2_2.append(ResBlock_256_2())
        self.ResBlock2_2 = nn.Sequential(*ResBlock2_2)

        ResBlock3_1 = []
        ResBlock3_1.append(ResBlock_512_1())
        self.ResBlock3_1 = nn.Sequential(*ResBlock3_1)

        ResBlock3_2 = []
        ResBlock3_2.append(ResBlock_512_2())
        self.ResBlock3_2 = nn.Sequential(*ResBlock3_2)

        self.C3_trans = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1)
        self.C2_trans = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.C1_trans = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1)

        self.C3_trans_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.C2_trans_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.C1_trans_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.upx2 = torch.nn.UpsamplingNearest2d(scale_factor=2)

        self.conv1_gra = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=5, stride=1)
        self.conv2_gra = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(44808, 800)
        self.fc2 = nn.Linear(800, 800)
        self.fc3 = nn.Linear(800, 1)

        self.fc1_w = nn.Linear(11208, 800)
        self.fc2_w = nn.Linear(800, 800)
        self.fc3_w = nn.Linear(800, 1)

    def forward(self, x):
        x_distort = x[0].view(-1, x[0].size(-3), x[0].size(-2), x[0].size(-1))
        x_gra = x[1].view(-1, x[1].size(-3), x[1].size(-2), x[1].size(-1))
        x_distance = x[2].view(-1, x[2].size(-3), x[2].size(-2), x[2].size(-1))
        x_ent = x[3].view(-1, x[3].size(-3), x[3].size(-2), x[3].size(-1))
        x_distance = x_distance.squeeze(3).squeeze(2)
        x_distance = x_distance.view(-1, self.num_flat_features(x_distance))
        x_ent = x_ent.squeeze(3).squeeze(2)
        x_ent = x_ent.view(-1, self.num_flat_features(x_ent))
        x_dis_ent = torch.cat((x_distance, x_ent), 1)

        h = self.conv0(x_distort)
        h = self.ResBlock1_1(h)
        h = F.relu(h)
        h = self.ResBlock1_2(h)
        h = F.relu(h)
        C1 = h

        h = self.ResBlock2_1(h)
        # print('ResBlock2_1 size:' + str(h.size()))#ResBlock2_1 size:torch.Size([128, 256, 8, 8])
        h = F.relu(h)
        h = self.ResBlock2_2(h)
        # print('ResBlock2_2 size:' + str(h.size()))#ResBlock2_2 size:torch.Size([128, 256, 8, 8])
        h = F.relu(h)
        C2 = h

        h = self.ResBlock3_1(h)
        h = F.relu(h)
        # print('ResBlock3_1 size:' + str(h.size()))#ResBlock3_1 size:torch.Size([128, 512, 4, 4])
        h = self.ResBlock3_2(h)
        h = F.relu(h)
        # print('ResBlock3_2 size:' + str(h.size()))#ResBlock3_2 size:torch.Size([128, 512, 4, 4])
        C3 = h

        C3 = self.C3_trans(C3)
        # print('C3 size:'+str(C3.size()))#C3 size:torch.Size([128, 128, 4, 4])
        C2 = self.C2_trans(C2)
        # print('C2 size:' + str(C2.size()))#C2 size:torch.Size([128, 128, 8, 8])
        C1 = self.C1_trans(C1)
        # print('C1 size:' + str(C1.size()))#C1 size:torch.Size([128, 128, 16, 16])

        C3x2 = self.upx2(C3)

        P3 = self.C3_trans_2(C3)
        # print('P3 size:' + str(P3.size()))#P3 size:torch.Size([128, 128, 2, 2])

        C2 = torch.add(C2, C3x2)
        C2x2 = self.upx2(C2)
        P2 = self.C2_trans_2(C2)
        # print('P2 size:' + str(P2.size()))#P2 size:torch.Size([128, 128, 6, 6])

        C1 = torch.add(C1, C2x2)
        P1 = self.C1_trans_2(C1)
        # print('P1 size:' + str(P1.size()))#P1 size:torch.Size([128, 128, 14, 14])



        P1_ = F.max_pool2d(P1, (2, 2), stride=2)
        P2_ = F.max_pool2d(P2, (2, 2), stride=2)
        P3_ = F.max_pool2d(P3, (2, 2), stride=2)

        P1_ = P1_.view(-1, self.num_flat_features(P1_))
        P2_ = P2_.view(-1, self.num_flat_features(P2_))
        P3_ = P3_.view(-1, self.num_flat_features(P3_))

        FP2 = torch.cat((P1_, P2_, P3_), 1)
        # print('FP2 size:' + str(FP2.size()))#[128, 10752]

        P3 = P3.view(-1, self.num_flat_features(P3))
        P2 = P2.view(-1, self.num_flat_features(P2))
        P1 = P1.view(-1, self.num_flat_features(P1))

        # print('P3 size:' + str(P3.size()))  # [128, 2048]
        # print('P2 size:' + str(P2.size()))  # [128, 8192]
        # print('P1 size:' + str(P1.size()))  # [128, 32768]

        FP1 = torch.cat((P1, P2, P3), 1)
        # print('FP1 size:'+str(FP1.size()))#[128, 43008]

        #梯度域
        x_gra = self.conv1_gra(x_gra)
        x_gra = F.max_pool2d(x_gra, (2, 2), stride=2)
        x_gra = self.conv2_gra(x_gra)
        x_gra = F.max_pool2d(x_gra, (2, 2), stride=2)
        x_gra = x_gra.squeeze(3).squeeze(2)

        FM2 = F.max_pool2d(x_gra, (1,2), stride=2)
        FM2 = FM2.view(-1, self.num_flat_features(FM2))
        # print('FM2 size:' + str(FM2.size()))#[128, 450]

        # print('x_gra size:'+str(x_gra.size()))#[128, 50, 6, 6]
        x_gra = x_gra.view(-1, self.num_flat_features(x_gra))
        # print('x_gra size:' + str(x_gra.size()))  #[128, 1800]

        s = torch.cat((FP1, x_gra), 1)
        # print('s size:'+str(s.size()))#[128, 44808]

        s = F.relu(self.fc1(s))
        s = F.dropout(s)
        s = F.relu(self.fc2(s))
        score_module = self.fc3(s)
        # print('score_module size:'+str(score_module.size()))#[128, 1]

        w = torch.cat((FP2, FM2), 1)
        w = torch.cat((w, x_dis_ent), 1)
        # print('w size:'+str(w.size()))#[128, 11208]

        w = F.relu(self.fc1_w(w))
        w = F.dropout(w)
        w = F.relu(self.fc2_w(w))
        weight_module = self.fc3_w(w)
        # print('weight_module size:' + str(weight_module.size()))  # [128, 1]

        q = score_module * weight_module

        return q


    def num_flat_features(self, xx):
        size = xx.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


class PyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, pyconv_kernels, pyconv_groups, stride=1, dilation=1, bias=False):
        super(PyConv2d, self).__init__()

        assert len(out_channels) == len(pyconv_kernels) == len(pyconv_groups)

        self.pyconv_levels = [None] * len(pyconv_kernels)
        for i in range(len(pyconv_kernels)):
            self.pyconv_levels[i] = nn.Conv2d(in_channels, out_channels[i], kernel_size=pyconv_kernels[i],
                                              stride=stride, padding=pyconv_kernels[i] // 2, groups=pyconv_groups[i],
                                              dilation=dilation, bias=bias)
        self.pyconv_levels = nn.ModuleList(self.pyconv_levels)

    def forward(self, x):
        out = []
        for level in self.pyconv_levels:
            out.append(level(x))

        return torch.cat(out, 1)


class CNNIQAnet(nn.Module):
    def __init__(self,ker_size=3, n_kers=64, n1_nodes=800, n2_nodes=800,conv=common.default_conv):

        super(CNNIQAnet, self).__init__()
        self.conv1 = nn.Conv2d(1, n_kers, 3,padding=1)
        m_body1 = [
            common.ResBlock2(
                conv, 128, 3,1, act=nn.ReLU(True)
            ) for _ in range(1)
        ]
        m_body11 = [
            common.ResBlock(
                conv, 128, 3, 1, act=nn.ReLU(True)
            ) for _ in range(1)
        ]
        m_body111 = [
            common.ResBlock(
                conv, 128, 3, 1, act=nn.ReLU(True)
            ) for _ in range(1)
        ]
        self.body1 = nn.Sequential(*m_body1)
        self.body11 = nn.Sequential(*m_body11)
        self.body111 = nn.Sequential(*m_body111)

        m_body2 = [
            common.ResBlock2(
                conv, 256, 3,1, act=nn.ReLU(True)
            ) for _ in range(1)
        ]
        m_body22 = [
            common.ResBlock(
                conv, 256, 3, 1, act=nn.ReLU(True)
            ) for _ in range(1)
        ]
        m_body222 = [
            common.ResBlock(
                conv, 256, 3, 1, act=nn.ReLU(True)
            ) for _ in range(1)
        ]
        self.body2 = nn.Sequential(*m_body2)
        self.body22 = nn.Sequential(*m_body22)
        self.body222 = nn.Sequential(*m_body222)


        m_body3 = [
            common.ResBlock2(
                conv, 512, 3,1, act=nn.ReLU(True)
            ) for _ in range(1)
        ]
        m_body33 = [
            common.ResBlock(
                conv, 512, 3, 1, act=nn.ReLU(True)
            ) for _ in range(1)
        ]
        m_body333 = [
            common.ResBlock(
                conv, 512, 3, 1, act=nn.ReLU(True)
            ) for _ in range(1)
        ]
        self.body3 = nn.Sequential(*m_body3)
        self.body33 = nn.Sequential(*m_body33)
        self.body333 = nn.Sequential(*m_body333)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d( 128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 128, 3, padding=1)
        self.convdel4=nn.Conv2d(512, 128, 1)
        self.convdel3=nn.Conv2d(256, 128, 1)
        self.convdel2=nn.Conv2d(128, 128, 1)
        self.conv11=nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,stride=1, padding=1)
        self.pyconv1 = PyConv2d(in_channels=16, out_channels=[16, 16, 32], pyconv_kernels=[3, 5, 7], pyconv_groups=[1, 4, 8])
        self.pyconv2 = PyConv2d(in_channels=64, out_channels=[16, 16, 32], pyconv_kernels=[3, 5, 7], pyconv_groups=[1, 4, 8])
        self.fc1 = nn.Linear(47104, n1_nodes)
        self.fc2 = nn.Linear(n1_nodes, n2_nodes)
        self.fc3 = nn.Linear(n2_nodes, 1)
        self.fc4 = nn.Linear(11782, n1_nodes)
        self.fc5 = nn.Linear(n1_nodes, n2_nodes)
        self.fc6 = nn.Linear(n2_nodes, 1)


    def forward(self, x):
        am = x[0].view(-1, x[0].size(-3), x[0].size(-2), x[0].size(-1))  # 32×32图片
        al = x[1].view(-1, x[1].size(-3), x[1].size(-2), x[1].size(-1))  # 32×32高频梯度图像，
        ai = x[2].view(-1, x[2].size(-3), x[2].size(-2), x[2].size(-1))  # 距离信息
        ae = x[3].view(-1, x[3].size(-3), x[3].size(-2), x[3].size(-1))

        hc1 = self.conv1(am)
        res2 = self.body1(hc1)
        hc2=F.relu(res2)
        res2 = self.body11(hc2)
        hc2 = F.relu(res2)
        res2 = self.body111(hc2)
        hc2 = F.relu(res2)

        res3 = self.body2(hc2)
        hc3=F.relu(res3)
        res3 = self.body22(hc3)
        hc3 = F.relu(res3)
        res3 = self.body222(hc3)
        hc3 = F.relu(res3)

        res4 = self.body3(hc3)
        hc4 = F.relu(res4)
        res4 = self.body33(hc4)
        hc4 = F.relu(res4)
        res4 = self.body333(hc4)
        hc4 = F.relu(res4)

        pc4=self.convdel4(hc4)
        m=torch.nn.UpsamplingNearest2d(scale_factor=2)
        pc3=m(pc4)
        c3=self.convdel3(hc3)
        p3=torch.add(pc3,c3)
        pc2=m(pc3)
        c2 = self.convdel2(hc2)
        p2 = torch.add(pc2, c2)
        p4 = self.conv5(pc4)
        p3 = self.conv6(p3)
        p2=self.conv7(p2)

        pp2= p2.view(-1, self.num_flat_features(p2))
        pp3= p3.view(-1, self.num_flat_features(p3))
        pp4= p4.view(-1, self.num_flat_features(p4))
        pp43 = torch.cat((pp4, pp3), 1)
        pp432 = torch.cat((pp43, pp2), 1)

        hl = self.conv11(al)
        hl = self.pyconv1(hl)
        hl = F.max_pool2d(hl, (2, 2), stride=2)
        hl = self.pyconv2(hl)
        hl = F.max_pool2d(hl, (2, 2), stride=2)
        hldel2 = F.max_pool2d(hl, (2, 2), stride=2)#
        hldel2 = hldel2.view(-1, self.num_flat_features(hldel2))
        h3 = hl.view(-1, self.num_flat_features(hl))
        hhhh=torch.cat((pp432,h3),1)
        p2del2=F.max_pool2d(p2, (2, 2), stride=2)
        p3del2 = F.max_pool2d(p3, (2, 2), stride=2)
        p4del2 = F.max_pool2d(p4, (2, 2), stride=2)
        pp2del2 = p2del2.view(-1, self.num_flat_features(p2del2))
        pp3del2 = p3del2.view(-1, self.num_flat_features(p3del2))
        pp4del2 = p4del2.view(-1, self.num_flat_features(p4del2))
        ppdel43 = torch.cat((pp4del2, pp3del2), 1)
        ppdel432 = torch.cat((ppdel43, pp2del2), 1)
        hhhdel2 = torch.cat((ppdel432, hldel2), 1)
        ai = ai.squeeze(3).squeeze(2)
        h4 = ai.view(-1, self.num_flat_features(ai))
        ae = ae.squeeze(3).squeeze(2)
        h5 = ae.view(-1, self.num_flat_features(ae))
        hhh1 = torch.cat((h4, h5), 1)

        hhh2 = torch.cat((hhh1, hhhdel2), 1)


        h = F.relu(self.fc1(hhhh))
        h = F.dropout(h)
        h = F.relu(self.fc2(h))
        qi = self.fc3(h)

        i = F.relu(self.fc4(hhh2))
        i = F.dropout(i)
        i = F.relu(self.fc5(i))
        w = self.fc6(i)

        q = qi * w
        return q

    def num_flat_features(self, xx):
        size = xx.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features








