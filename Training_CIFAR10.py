# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 10:39:25 2021

@author: admin
"""

import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
# import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
# from Tools import filters, JSMA
from torchvision import utils as vutils
import numpy as np
from collections import OrderedDict
import random
import advertorch
from advertorch.attacks import LinfPGDAttack, CarliniWagnerL2Attack, DDNL2Attack, SinglePixelAttack, LocalSearchAttack, SpatialTransformAttack,L1PGDAttack
from autoattack import AutoAttack
from models.LeNet import LeNet5_CIFAR10
from models.ResNet import ResNet18_cifar, ResNet_autoencoder, ResidualBlock, Classifier_head, ResNet_Encoder, Decoder, ResNet18_cifar_dropout
from models.vgg import vgg
import torchvision.transforms as transforms


# basic settings
seed = 5
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
NUM_EPOCHS = 120
LEARNING_RATE = 1e-2
BATCH_SIZE = 512


###preprocess###
transform = transforms.Compose(
    [transforms.Resize([32, 32]),
        transforms.ToTensor()
     ]
)

trainset = datasets.CIFAR10(
    root='/media/cqu/D/FXV/PSSR_master',
    train=True,
    download=True,
    transform=transform
)
testset = datasets.CIFAR10(
    root='/media/cqu/D/FXV/PSSR_master',
    train=False,
    download=True,
    transform=transform
)


trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
testloader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False
)



def compute_kl_loss( p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


model = vgg(16, 'cifar10').cuda().train()


optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))
criterion_cls = nn.CrossEntropyLoss()

train_loss = []
b = 0.1
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for img, label in trainloader:
        img, label = img.cuda(), label.cuda()
        optimizer.zero_grad()

        logits1 = model(img)
        logits2 = model(img)

        ce_loss = criterion_cls(logits1, label) + criterion_cls(logits2, label)
        kl_loss = compute_kl_loss(logits1, logits2)
        loss = ce_loss #+ 0.07 * kl_loss

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    loss = running_loss / len(trainloader)
    train_loss.append(loss)
    print('Epoch {} of {}, Train Loss: {:.6f}, Cls Loss: {:.6f}, KL Loss: {:.6f},'.format(epoch + 1, NUM_EPOCHS, loss, ce_loss, kl_loss))

    torch.save(model, './saving_models/CIFAR10/VGG16.pkl')
#












# # # test
# total = 0
# correct = 0
# temp = 0
# test_model = torch.load('./saving_models/demo_try/ResNet_CIFAR10.pkl').eval()
# targetlabel = torch.zeros(BATCH_SIZE).cuda()
# targetlabel = targetlabel.to('cuda', dtype=torch.int64)
# for img, label in testloader:
#     length = img.shape[0]
#     targetlabel_temp = targetlabel[0:length]
#     img, label = img.cuda(), label.cuda()
#
#     img_adv = img
#     # img_adv = FGSM_N.perturb(img)
#     # img_adv = PGD_T.perturb(img, targetlabel_temp)
#     # img_adv = AA_N.run_standard_evaluation(img, label, bs=BATCH_SIZE)
#
#     x = test_model(img_adv)
#     _, prediction = torch.max(x, 1)
#     total += label.size(0)
#     correct += (prediction == label).sum()
#     # vutils.save_image(img_adv, './saving_samples/AA_N/img_adv_{}.jpg'.format(temp))
#     print('当前temp:', temp, '当前batch正确的个数:', (prediction == label).sum())
#     temp += 1
# print('There are ' + str(correct.item()) + ' correct pictures.')
# print('Accuracy=%.2f' % (100.00 * correct.item() / total))





