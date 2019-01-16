import torch.nn.functional as F


def inception_forward(inception_v3, image):
    if inception_v3.transform_input:
        image[:, 0] = image[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        image[:, 1] = image[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        image[:, 2] = image[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
    # 299 x 299 x 3
    image = inception_v3.Conv2d_1a_3x3(image)
    # 149 x 149 x 32
    image = inception_v3.Conv2d_2a_3x3(image)
    # 147 x 147 x 32
    image = inception_v3.Conv2d_2b_3x3(image)
    # 147 x 147 x 64
    image = F.max_pool2d(image, kernel_size=3, stride=2)
    # 73 x 73 x 64
    image = inception_v3.Conv2d_3b_1x1(image)
    # 73 x 73 x 80
    image = inception_v3.Conv2d_4a_3x3(image)
    # 71 x 71 x 192
    image = F.max_pool2d(image, kernel_size=3, stride=2)
    # 35 x 35 x 192
    image = inception_v3.Mixed_5b(image)
    # 35 x 35 x 256
    image = inception_v3.Mixed_5c(image)
    # 35 x 35 x 288
    image = inception_v3.Mixed_5d(image)
    # 35 x 35 x 288
    image = inception_v3.Mixed_6a(image)
    # 17 x 17 x 768
    image = inception_v3.Mixed_6b(image)
    # 17 x 17 x 768
    image = inception_v3.Mixed_6c(image)
    # 17 x 17 x 768
    image = inception_v3.Mixed_6d(image)
    # 17 x 17 x 768
    image = inception_v3.Mixed_6e(image)
    # 17 x 17 x 768
    image = inception_v3.Mixed_7a(image)
    # 8 x 8 x 1280
    image = inception_v3.Mixed_7b(image)
    # 8 x 8 x 2048
    image = inception_v3.Mixed_7c(image)
    # 8 x 8 x 2048
    return image