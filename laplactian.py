import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

import torch
from PIL import Image
import torchvision.transforms as transforms

class GaussianFilter(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0):
        super(GaussianFilter, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([TF.gaussian_blur(img, kernel_size=self.kernel_size, sigma=self.sigma) for img in x])


# 1. 이미지 파일 불러오기
image_path = '/home/minmaxhong/catkin_ws/src/mmLab/IR1.png'  # 이미지 파일 경로
image = Image.open(image_path).convert('L')

# 2. 이미지를 PyTorch 텐서로 변환
transform = transforms.Compose([
    transforms.ToTensor(),  # 이미지에서 텐서로 변환
    transforms.Normalize(mean=[0.5], std=[0.5])  # 정규화
])
image_tensor = transform(image).unsqueeze(0)  # 배치 차원 추가

# 3. 가우시안 필터 클래스 적용
gaussian_filter = GaussianFilter(kernel_size=5, sigma=1.5)
output_tensor = gaussian_filter(image_tensor)

# 4. 텐서를 PIL 이미지로 변환해 확인
output_image = transforms.ToPILImage()(output_tensor.squeeze(0))
output_image.show()  # 결과 이미지 확인
