import torch
import torch.nn as nn
import cv2
import numpy as np
from kornia.filters import GaussianBlur2d, Laplacian  # GaussianBlur2d와 Laplacian 임포트
from torch import Tensor


class GaussianEdgeDetect(nn.Module):
    def __init__(self):
        super(GaussianEdgeDetect, self).__init__()
        self.gaussian_blur = GaussianBlur2d(kernel_size=(5, 5), sigma=(1.5, 1.5))  # Gaussian 블러 설정
        self.laplacian = Laplacian(kernel_size=3)  # Laplacian 필터 설정

    def forward(self, x: Tensor) -> Tensor:
        x_blurred = self.gaussian_blur(x)  # Gaussian 블러 적용
        edges = self.laplacian(x_blurred)  # Laplacian 적용
        edges = torch.abs(edges)
        return edges

def main():
    vis_path = "/home/minmaxhong/Dataset/KAIST/VIS1.png"
    
    # 이미지 로드 및 채널 변환
    img = cv2.imread(vis_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR 형식으로 이미지를 읽음
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (H, W, C) -> (C, H, W)로 변환 및 정규화

    edge_detector = GaussianEdgeDetect()
    edge_img_tensor = edge_detector(img_tensor.unsqueeze(0))  # 배치 차원 추가

    # 텐서를 NumPy 배열로 변환하여 OpenCV에서 사용
    edge_img = edge_img_tensor.squeeze(0).permute(1, 2, 0).detach().numpy()  # (B, C, H, W) -> (H, W, C)

    # 최대값과 최소값을 기준으로 정규화
    edge_img = (edge_img - edge_img.min()) / (edge_img.max() - edge_img.min())  # [0, 1] 범위로 정규화

    # 값을 [0, 255] 범위로 변환
    edge_img = (edge_img * 255).astype(np.uint8)

    # 이미지 표시
    cv2.imshow('Original Image', img)
    cv2.imshow('Edge Detected Image', edge_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
