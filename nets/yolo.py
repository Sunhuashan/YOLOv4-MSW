from collections import OrderedDict
from turtle import width
from matplotlib.pyplot import axis

import torch
import torch.nn as nn
from .asff import ASFF
from .rfb  import BasicRFB as RFB

from nets.CSPdarknet import darknet53


def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#---------------------------------------------------#
#   SPP结构，利用不同大小的池化核进行池化
#   池化后堆叠
#---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features

#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

#---------------------------------------------------#
#   下采样
#---------------------------------------------------#
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()

        self.downsample = nn.Sequential(
            nn.MaxPool2d(2,2,0)
        )

    def forward(self, x,):
        x = self.downsample(x)
        return x

#---------------------------------------------------#
#   三次卷积块
#---------------------------------------------------#
def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   五次卷积块
#---------------------------------------------------#
def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes):
        super(YoloBody, self).__init__()
        #---------------------------------------------------#   
        #   生成CSPdarknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #   104,104,128
        #---------------------------------------------------#
        self.backbone = darknet53(None)

        self.conv1      = make_three_conv([512,1024],1024)
        self.SPP        = SpatialPyramidPooling()
        self.conv2      = make_three_conv([512,1024],2048)

        # rfb
        self.rfb1=RFB(in_planes=128,out_planes=64)
        self.rfb2=RFB(in_planes=1024,out_planes=512)
        self.upsample_for_p2=Upsample(128,64)
        # self.conv1_for_p2=make_three_conv([64,128],128)
        # self.conv2_for_p2=make_three_conv([64,256],256)
        # concat
        self.make_five_conv_for_p2_down=make_five_conv([64,128],128)
        self.down_sample_for_p2=conv2d(64,128,3,stride=2)
        self.make_five_conv_for_P3=make_five_conv([128,256],256)


        

        self.upsample1          = Upsample(512,256)
        self.conv_for_P4        = conv2d(512,256,1)
        self.make_five_conv1    = make_five_conv([256, 512],512)

        self.upsample2          = Upsample(256,128)
        self.conv_for_P3        = conv2d(256,128,1)
        self.make_five_conv2    = make_five_conv([128, 256],256)

        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        self.yolo_head3         = yolo_head([256, len(anchors_mask[0]) * (5 + num_classes)],128)

        self.down_sample1       = conv2d(128,256,3,stride=2)
        self.make_five_conv3    = make_five_conv([256, 512],512)

        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        self.yolo_head2         = yolo_head([512, len(anchors_mask[1]) * (5 + num_classes)],256)

        self.down_sample2       = conv2d(256,512,3,stride=2)
        self.make_five_conv4    = make_five_conv([512, 1024],1024)

        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        self.yolo_head1         = yolo_head([1024, len(anchors_mask[2]) * (5 + num_classes)],512)

        self.asff_0=ASFF(level=0, multiplier = 0.5)
        self.asff_1=ASFF(level=1, multiplier = 0.5)
        self.asff_2=ASFF(level=2, multiplier = 0.5)

    def forward(self, x):
        #  backbone
        x2, x1, x0 ,temp= self.backbone(x)   # temp为DownSample输出,104x104x128
        # x2, x1, x0= self.backbone(x)    


        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,2048 
        # P5 = self.conv1(x0)
        # P5 = self.SPP(P5)
        # 13,13,2048 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        # P5 = self.conv2(P5)

        # rfb 2
        P5=self.rfb2(x0)

        #extra module
        # P2=self.down_sample1_for_P2(temp)
        


        # 13,13,512 -> 13,13,256 -> 26,26,256
        P5_upsample = self.upsample1(P5)
        # 26,26,512 -> 26,26,256
        P4 = self.conv_for_P4(x1)
        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = torch.cat([P4,P5_upsample],axis=1)
        # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        P4 = self.make_five_conv1(P4)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        P4_upsample = self.upsample2(P4)
        # 52,52,256 -> 52,52,128
        P3 = self.conv_for_P3(x2)

        #fuse P3 and P2
        # P3=torch.cat([P2,P3],axis=1)
        # P3=self.make_five_conv0(P3)


        # 52,52,128 + 52,52,128 -> 52,52,256
        P3 = torch.cat([P3,P4_upsample],axis=1)
        # 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        P3 = self.make_five_conv2(P3)

        
        #extra for RFB
        P2=self.rfb1(temp)   #104x104x128->104x104x64
        # P2=self.conv1_for_p2(temp)  #104x104x128 -> 104x104x64
        # P2=self.SPP(P2)  #104x104x64 ->104x104x256
        # P2=self.conv2_for_p2(P2) # 104x104x256 ->104x104x64

        P3_upsample=self.upsample_for_p2(P3)  #52x52x128->104x104x64
        P2=torch.cat([P2,P3_upsample],axis=1)  #104x104x128

        P2=self.make_five_conv_for_p2_down(P2)  #104x104x128 ->104x104x64
        P2=self.down_sample_for_p2(P2)    #104x104x64->52x52x128
        P3=torch.cat([P3,P2],axis=1)   # 52x52x256
        P3=self.make_five_conv_for_P3(P3)  #52x52x128

# ----------------------------------------------------------------------------------
        # 52,52,128 -> 26,26,256
        P3_downsample = self.down_sample1(P3)
        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = torch.cat([P3_downsample,P4],axis=1)
        # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        P4 = self.make_five_conv3(P4)
        # P4=torch.add(P4,init_P4) #横向融合 step1    


        # 26,26,256 -> 13,13,512
        P4_downsample = self.down_sample2(P4)
        # 13,13,512 + 13,13,512 -> 13,13,1024
        P5 = torch.cat([P4_downsample,P5],axis=1)
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        P5 = self.make_five_conv4(P5)


        outputs=(P3,P4,P5)
        pan_out0=self.asff_0(outputs)
        pan_out1=self.asff_1(outputs)
        pan_out2=self.asff_2(outputs)
        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,75,52,52)
        #---------------------------------------------------#
        out2=self.yolo_head3(pan_out2)
        # out2 = self.yolo_head3(P3)
        #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,75,26,26)
        #---------------------------------------------------#
        out1=self.yolo_head2(pan_out1)
        # out1 = self.yolo_head2(P4)
        # ---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,75,13,13)
        #---------------------------------------------------#
        out0=self.yolo_head1(pan_out0)
        # out0 = self.yolo_head1(P5)

        return out0, out1, out2

