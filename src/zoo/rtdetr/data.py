import torch
import numpy as np
from skimage.feature import canny
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from torchvision.transforms import ToTensor, ToPILImage
from skimage.color import rgb2gray

def tensor_to_image():
    return ToPILImage()

def image_to_tensor():
    return ToTensor()

def image_to_edge(images, sigma):
    # 输入 images 的形状为 [b, c, h, w]
    batch_size = images.size(0)
    edges = []
    gray_images = []
    
    for i in range(batch_size):
        # 取出每个图像，形状为 [c, h, w]
        image = images[i]
        # 将图像转换为灰度图像
        gray_image = rgb2gray(np.array(tensor_to_image()(image)))
        # 计算边缘图像
        edge = image_to_tensor()(Image.fromarray(canny(gray_image, sigma=sigma)))
        gray_image = image_to_tensor()(Image.fromarray(gray_image))
        # 将结果加入列表
        edges.append(edge)
        gray_images.append(gray_image)

    # 将列表转换为张量，形状为 [b, 1, h, w]
    edges = torch.stack(edges, dim=0)
    gray_images = torch.stack(gray_images, dim=0)

    return edges
