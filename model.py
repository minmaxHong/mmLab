import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
import cv2
import numpy as np

from kornia.filters import SpatialGradient
from torch import Tensor

# IR/VIS image shape: [270(height), 360(width), 3(channel)] -> train에서는 gray scale로 channel: 1

# Edge Detection
class EdgeDetect(nn.Module):
    def __init__(self):
        super(EdgeDetect, self).__init__()
        self.spatial = SpatialGradient('diff') # input image의 각 x, y축에 대해 미분 계산
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
    def forward(self, x: Tensor) -> Tensor:
        s = self.spatial(x)
        dx, dy = s[:, :, 0, :, :], s[:, :, 1, :, :] # [Batch size, Channel, 2(x축과 y축의 미분 index), Height, Width]
        magintude = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2)) # edge strength
        filtering_magintude = self.max_pool(magintude) # edge가 큰 값만 extraction
        
        return filtering_magintude


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()
        self.edge = EdgeDetect()

    # source images의 input pixel은 [-1, 1]이다.
    def forward(self, ir_img, rgb_img, max_pixel=255, epsilon=0.00001): # ir_img, rgb_img:  [-1, 1]의 input값 들어옴
        ir_attention = ((ir_img * (max_pixel / 2)) + max_pixel / 2) / max_pixel
        rgb_attention = ((rgb_img * (max_pixel / 2)) + max_pixel / 2) / max_pixel
        
        ir_edgemap = self.edge(ir_attention)
        rgb_edgemap = self.edge(rgb_attention)
        
        edgemap_ir = ir_edgemap / (ir_edgemap + rgb_edgemap + epsilon) # 0.0001은 0으로 나누어지는 것을 방지하기 위함
        edgemap_ir = (edgemap_ir - 0.5) * 2
        
        edgemap_rgb = rgb_edgemap / (ir_edgemap + rgb_edgemap + epsilon)
        edgemap_rgb = (edgemap_rgb - 0.5) * 2
        
        return edgemap_ir, edgemap_rgb

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1):
        super(ConvLayer, self).__init__()

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
    
    def forward(self, x):
        out = self.conv2d(x)
        return out

class DownConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DownConvLayer, self).__init__()

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        out = self.conv2d(x)
        out = self.gelu(out)
        return out

# Residual Feature Distillation Block
class RFDB(nn.Module):
    def __init__(self, _1x1_conv=1, _3x3_conv=3, in_channels=32, distillation_rate=0.25):
        super(RFDB, self).__init__()
        self.dc = self.distilled_channels = in_channels // 2 # dim: 16
        self.rc = self.remaining_channels = in_channels # dim: 32

        self.c1_d = nn.Conv2d(in_channels, self.dc, kernel_size=_1x1_conv, padding=0)
        self.c2_d = nn.Conv2d(in_channels, self.dc, kernel_size=_1x1_conv, padding=0)

        self.c1_r = nn.Conv2d(in_channels, self.rc, kernel_size=_3x3_conv, padding=1)
        self.c2_r = nn.Conv2d(in_channels, self.rc, kernel_size=_3x3_conv, padding=1)
        

        self.gelu = nn.GELU()

    def forward(self, x):
        distilled_c1 = self.gelu(self.c1_d(x)) # dim: [Batch, 16, 68, 90]
        r_c1 = self.c1_r(x)
        r_c1 = self.gelu(r_c1 + x) # dim: [Batch, 32, 68, 90]

        distilled_c2 = self.gelu(self.c2_d(r_c1)) # dim: [Batch, 16, 68, 90]
        print('distilled c2: ', distilled_c2.shape)
        r_c2 = self.c2_r(r_c1) 
        r_c2 = self.c2_r(r_c2 + r_c1) # dim: [Batch, 32, 68, 90]  
        print('r_c2:' , r_c2.shape)


        return r_c1




class CMTFusion(nn.Module):
    def __init__(self):
        super(CMTFusion, self).__init__()
        self.normalize = Normalize()

        kernel_size = 3
        stride = 1
        nb_filter = [32, 32, 48, 64] # number of filter ~~ 
        _1x1_kernel = 1
        padding = 1

        down_in_out_channels = 32
        down_kernel_size = 3
        down_stride = 2
        down_padding = 1

        self.conv_ir1 = ConvLayer(2, nb_filter[0], kernel_size, stride, padding)
        self.conv_rgb1 = ConvLayer(2, nb_filter[0], kernel_size, stride, padding)

        self.conv_ir_1x1_level1 = ConvLayer(nb_filter[0], nb_filter[0], _1x1_kernel, stride, 0)
        self.conv_ir_1x1_level2 = ConvLayer(nb_filter[0], nb_filter[0], _1x1_kernel, stride, 0)
        self.conv_ir_1x1_level3 = ConvLayer(nb_filter[0], nb_filter[0], _1x1_kernel, stride, 0)

        self.conv_rgb_1x1_level1 = ConvLayer(nb_filter[0], nb_filter[0], _1x1_kernel, stride, 0)
        self.conv_rgb_1x1_level2 = ConvLayer(nb_filter[0], nb_filter[0], _1x1_kernel, stride, 0)
        self.conv_rgb_1x1_level3 = ConvLayer(nb_filter[0], nb_filter[0], _1x1_kernel, stride, 0)

        self.ir_down1 = DownConvLayer(down_in_out_channels, down_in_out_channels, down_kernel_size, down_stride, down_padding)
        self.ir_down2 = DownConvLayer(down_in_out_channels, down_in_out_channels, down_kernel_size, down_stride, down_padding)

        self.vis_down1 = DownConvLayer(down_in_out_channels, down_in_out_channels, down_kernel_size, down_stride, down_padding)
        self.vis_down2 = DownConvLayer(down_in_out_channels, down_in_out_channels, down_kernel_size, down_stride, down_padding)

        self.ir_encoder_level3 = RFDB()


    def forward(self, rgb, ir):
        # == Preprocessing == 
        edgemap_ir, edgemap_rgb = self.normalize(ir, rgb) # dim: [Batch, 1, 270, 360]

        # channel-wise로 concat
        ir_input = torch.cat([ir, edgemap_ir], dim=1) # dim: [Batch, 2, 270, 360]
        vis_input = torch.cat([rgb, edgemap_rgb], dim=1)
        # ===============================================

        # L-Level의 feature pyramid 구성
        ir_level1 = self.conv_ir1(ir_input) # dim: [Batch, 32, 270, 360]
        ir_level2 = self.ir_down1(ir_level1) # dim: [Batch, 32, 135, 180]
        ir_level3 = self.ir_down2(ir_level2) # dim: [Batch, 32, 68, 90]

        rgb_level1 = self.conv_rgb1(vis_input) # same, ir_level1
        rgb_level2 = self.vis_down1(rgb_level1) # same, ir_level2
        rgb_level3 = self.vis_down2(rgb_level2) # same, ir_level3

        ir_level3_1x1 = self.conv_ir_1x1_level3(ir_level3) # dim: [Batch, 32, 68, 90]
        rgb_level3_1x1 = self.conv_rgb_1x1_level3(rgb_level3) # dim: [Batch, 32, 68, 90]
        
        ir_level2_1x1 = self.conv_ir_1x1_level2(ir_level2) # dim: [Batch, 32, 135, 180]
        rgb_level2_1x1 = self.conv_rgb_1x1_level2(rgb_level2) # dim: [Batch, 32, 135, 180]

        ir_level1_1x1 = self.conv_ir_1x1_level1(ir_level1) # dim: [Batch, 32, 270, 360]
        rgb_level1_1x1 = self.conv_rgb_1x1_level1(rgb_level1) # dim: [Batch, 32, 270, 360]
        # ===============================================

        # L-Level의 feature pyramid를 RFDB에 넣어, discriminativ한 feature map 구성
        ir_level3_rfdb = self.ir_encoder_level3(ir_level3_1x1)


def process_image_with_edge_detection(rgb_path, ir_path):
    rgb_image = cv2.imread(rgb_path) # [height, width, channel]
    ir_image = cv2.imread(ir_path) 

    # BGR -> GRAY
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    ir_image = cv2.cvtColor(ir_image, cv2.COLOR_BGR2GRAY)

    # np -> tensor
    rgb_image_tensor = torch.tensor(rgb_image)
    rgb_image_tensor = rgb_image_tensor.unsqueeze(0).unsqueeze(0)

    ir_image_tensor = torch.tensor(ir_image)
    ir_image_tensor = ir_image_tensor.unsqueeze(0).unsqueeze(0)

    # [-1, 1] 정규화
    rgb_image_tensor = (rgb_image_tensor.to(torch.float32) / 255.0) * 2 - 1 
    ir_image_tensor = (ir_image_tensor.to(torch.float32) / 255.0) * 2 - 1

    cmtfusion = CMTFusion()
    cmtfusion.forward(rgb_image_tensor, ir_image_tensor)

rgb_path = r"C:\Users\minmaxHong\Desktop\code\minmaxHong_github\mmLab\VIS1.png"
ir_path = r"C:\Users\minmaxHong\Desktop\code\minmaxHong_github\mmLab\IR1.png"

process_image_with_edge_detection(rgb_path, ir_path)