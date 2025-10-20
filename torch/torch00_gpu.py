import torch

print('Pytorch 버전 :', torch.__version__)

cuda_available = torch.cuda.is_available()
print('CUDA 사용 가능 여부 :', cuda_available)

gpu_count = torch.cuda.device_count()
print('사용 가능 GPU 갯수 :', gpu_count)

print(torch.cuda.get_device_capability(0)) 

if cuda_available:
    current_device = torch.cuda.current_device()
    print("현재 사용중인 GPU 장치 ID :", current_device)
    print("현재 GPU 이름 :", torch.cuda.get_device_name(current_device))
else :
    print("GPU 없음")
    
print("CUDA 버전 :", torch.version.cuda)

cudnn_version = torch.backends.cudnn.version()
if cudnn_version is not None :
    print("CuDNN 버전 :", cudnn_version)
else:
    print("CuDNN 없음")
    
# Pytorch 버전 : 2.7.1+cu128
# CUDA 사용 가능 여부 : True
# 사용 가능 GPU 갯수 : 1
# (12, 0)
# 현재 사용중인 GPU 장치 ID : 0
# 현재 GPU 이름 : NVIDIA GeForce RTX 5070 Ti
# CUDA 버전 : 12.8
# CuDNN 버전 : 90701


# 1. epoch 없음! for문 돌려야 함
# 2. DataLoader 라는 게 있음
# 3. Class 정의를 해야함

