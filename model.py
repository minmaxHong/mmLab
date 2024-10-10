import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
import cv2
import numpy as np

from kornia.filters import SpatialGradient
from torch import Tensor

# IR/VIS image shape: [270(height), 360(width), 3(channel)]


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

# =====================================================================
# 질문 !
class attention(nn.Module):
    def __init__(self):
        super(attention, self).__init__()
        self.edge = EdgeDetect()

    def forward(self, ir_img, rgb_img):
        ir_attention = ((ir_img * 127.5) + 127.5) / 255
        rgb_attention = ((rgb_img * 127.5) + 127.5) / 255
        
        ir_edgemap = self.ed(ir_attention)
        rgb_edgemap = self.ed(rgb_attention)
        
        edgemap_ir = ir_edgemap / (ir_edgemap + rgb_edgemap + 0.00001)
        edgemap_ir = (edgemap_ir - 0.5) * 2
        
        edgemap_rgb = rgb_edgemap / (ir_edgemap + rgb_edgemap + 0.00001)
        edgemap_rgb = (edgemap_rgb - 0.5) * 2
        
        return edgemap_ir, edgemap_rgb
    
# =====================================================================
    
        
def process_image_with_edge_detection(image_path):
    image = cv2.imread(image_path) # [height, width, channel]
    
    
image_path = "VIS1.png"
process_image_with_edge_detection(image_path)
