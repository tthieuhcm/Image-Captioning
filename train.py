import torch
import torchvision
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from tqdm import tqdm

from Inception_conv import inception_forward
from dataloader import MSCOCO_Dataset

# We'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
# import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
# from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle
#
# import re
import numpy as np

# import os
# import time
# import json
# from glob import glob
# from PIL import Image
# import pickle

root_dir = './data/train2014/'
annotation_file = './data/captions_train2014.json'

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# selecting the first 30000 captions from the set
num_examples = 30000

model_conv = torchvision.models.inception_v3(pretrained=True)

# model_conv = model_conv.to(device)

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor()])

MSCOCO_dataset = MSCOCO_Dataset(annotation_file=annotation_file,
                                root_dir=root_dir,
                                transform=transform,
                                num_examples=num_examples)

MSCOCO_dataloader = torch.utils.data.DataLoader(MSCOCO_dataset, batch_size=8,
                                                shuffle=True, num_workers=4)

model_conv.eval()
with torch.no_grad():
    for sample in tqdm(MSCOCO_dataloader):
        batch_features = inception_forward(model_conv, sample['image'])
        batch_features = batch_features.view(batch_features.shape[0], -1, batch_features.shape[1])

        for bf, p in zip(batch_features, sample['path']):
            np.save(p, bf.numpy())
#
#
#
# import torch
# import torchvision
# # import torch.nn as nn
# # import torch.optim as optim
# # from torch.optim import lr_scheduler
# from torchvision import datasets, models, transforms
# from tqdm import tqdm
#
# from Inception_conv import inception_forward
# from dataloader import MSCOCO_Dataset
#
# # We'll generate plots of attention in order to see which parts of an image
# # our model focuses on during captioning
# # import matplotlib.pyplot as plt
#
# # Scikit-learn includes many helpful utilities
# # from sklearn.model_selection import train_test_split
# # from sklearn.utils import shuffle
# #
# # import re
# import numpy as np
#
# # import os
# # import time
# # import json
# # from glob import glob
# # from PIL import Image
# # import pickle
#
# root_dir = './data/train2014/'
# annotation_file = './data/captions_train2014.json'
#
# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# # selecting the first 30000 captions from the shuffled set
# num_examples = 30000
#
# model_conv = torchvision.models.inception_v3(pretrained=True)
#
# # model_conv = model_conv.to(device)
#
# data_transforms = transforms.Compose([transforms.Resize((299, 299)),
#                                       transforms.ToTensor()])
#
# MSCOCO_dataset = datasets.CocoCaptions(root=root_dir,
#                                        annFile=annotation_file,
#                                        transform=data_transforms)
#
# MSCOCO_dataloader = torch.utils.data.DataLoader(MSCOCO_dataset, batch_size=16,
#                                                 shuffle=True, num_workers=4)
#
# model_conv.eval()
# with torch.no_grad():
#     for (images, captions) in tqdm(MSCOCO_dataloader):
#         batch_features = inception_forward(model_conv, images)
#         batch_features = batch_features.view(batch_features.shape[0], -1, batch_features.shape[1])
#
#         for bf, p in zip(batch_features, sample['path']):
#             np.save(p, bf.numpy())
