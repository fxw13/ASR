# 这是一个示例 Python 脚本。

import torch
# import cv2
# import pytorch_fft.fft as fft
# from PIL import Image
# from torchvision import utils as vutils
# import torchvision.transforms as transforms
# from torch.autograd import Variable
# import os

import torchvision.transforms as T
import copy

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
# from Tools import filters, JSMA,
from torchvision import utils as vutils
import numpy as np
from collections import OrderedDict
import random
# from frank_wolfe import FrankWolfe
# from autoattack import AutoAttack
import advertorch
from advertorch.attacks import LinfPGDAttack, CarliniWagnerL2Attack, DDNL2Attack, SinglePixelAttack, LocalSearchAttack, SpatialTransformAttack,L1PGDAttack
from autoattack import AutoAttack
from models.LeNet import LeNet5_autoencoder, LeNet5_encoder, Decoder, Classifier_head, LeNet5, LeNet5_STA, LeNet5_tsne, LeNet5_DA, Discriminator, LeNet5_AutoEncoder_CIFAR10
from models.ResNet import ResNet18_MNIST, ResidualBlock
import torchvision.transforms as transforms
# import pytorch_fft.fft as fft
from models.ResNet import ResNet18_cifar, ResNet_autoencoder, ResidualBlock, Classifier_head, ResNet_Encoder, Decoder, ResNet18_DA, ResNet18_Denoiser
from tools.transforms import RandomErasing, RandomErasing_adv


# FFT2D = fft.Fft2d()
# IFFT2D = fft.Ifft2d()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 0
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
NUM_EPOCHS = 90
LEARNING_RATE = 1e-3
BATCH_SIZE = 128

class SamplerDef(object):

    def __init__(self, data_source, indices):
        self.data_source = data_source
        self.indices = indices

    def __iter__(self):
        print(iter(self.indices))
        return iter(self.indices)

    def __len__(self):
        return len(self.data_source)

transform = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)

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

n = len(trainset)
indices = torch.randperm(n)
mySampler = SamplerDef(data_source=trainset, indices=indices)
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

class EMA(object):
    def __init__(self, model, alpha=0.999, buffer_ema=True):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha
        self.buffer_ema = buffer_ema
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self, model, model2=None, beta=None):
        decay = min(self.alpha, (self.step + 1) / (self.step + 10))
        state = model.state_dict()
        for i, name in enumerate(self.param_keys):
            if model2 and beta:
                if i < len(self.param_keys):
                    state2 = model2.state_dict()
                    self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * (state[name] * beta + state2[name] * (1-beta)))
            else:
                self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * state[name])
        for i, name in enumerate(self.buffer_keys):
            if self.buffer_ema:
                if model2 and beta:
                    if i < len(self.buffer_keys):
                        state2 = model2.state_dict()
                        self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * (state[name] * beta + state2[name] * (1-beta)))
                else:
                    self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * state[name])
            else:
                self.shadow[name].copy_(state[name])
        self.step += 1

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }

target_model = torch.load('/media/cqu/D/FXV/PSSR_master/saving_models/CIFAR10/VGG16_CIFAR10.pkl')
target_model.eval()

# # pixel constrain
FGSM_N = advertorch.attacks.GradientSignAttack(predict=target_model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.03)

FGSM_T = advertorch.attacks.GradientSignAttack(predict=target_model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.03,
targeted=True)

MMT_N = advertorch.attacks.MomentumIterativeAttack(predict=target_model, loss_fn=nn.CrossEntropyLoss(), eps=0.3, nb_iter=40,
                                                   decay_factor=1.0, eps_iter=0.01,
                                                   clip_min=0.0, clip_max=1.0, targeted=False)
MMT_T = advertorch.attacks.MomentumIterativeAttack(predict=target_model, loss_fn=nn.CrossEntropyLoss(), eps=0.3, nb_iter=40,
                                                   decay_factor=1.0, eps_iter=0.01,
                                                   clip_min=0.0, clip_max=1.0, targeted=True)
BIM_N = advertorch.attacks.LinfBasicIterativeAttack(predict=target_model, loss_fn=nn.CrossEntropyLoss(), eps=0.03, nb_iter=40,
                                                    eps_iter=0.05,
                                                    clip_min=0.0, clip_max=1.0, targeted=False)
BIM_T = advertorch.attacks.LinfBasicIterativeAttack(predict=target_model, loss_fn=nn.CrossEntropyLoss(), eps=0.3, nb_iter=40,
                                                    eps_iter=0.05,
                                                    clip_min=0.0, clip_max=1.0, targeted=True)

PGD_N = LinfPGDAttack(
            target_model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.030,
            nb_iter=100, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)

PGD_T = LinfPGDAttack(
            target_model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.030,
            nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)


CW_N = CarliniWagnerL2Attack(
    target_model, 10, clip_min=0.0, clip_max=1.0, max_iterations=500, confidence=1, initial_const=1, learning_rate=1e-2,
    binary_search_steps=4, targeted=False)

CW_T = CarliniWagnerL2Attack(
    target_model, 10, clip_min=0.0, clip_max=1.0, max_iterations=500, confidence=1, initial_const=1, learning_rate=1e-2,
    binary_search_steps=4, targeted=True)

DDN = DDNL2Attack(target_model, nb_iter=100, gamma=0.05, init_norm=1.0, quantize=True, levels=256, clip_min=0.0,
                        clip_max=1.0, targeted=False, loss_fn=None)

STA = SpatialTransformAttack(
    target_model, 10, clip_min=0.0, clip_max=1.0, max_iterations=5000, search_steps=20, targeted=False)

AA_N = AutoAttack(target_model, norm='Linf', eps=0.03, version='standard')

JSMA_T = advertorch.attacks.JacobianSaliencyMapAttack(predict=target_model, num_classes=10, clip_min=0.0, clip_max=1.0,
                                                    loss_fn=None, theta=1.0, gamma=1.0,
                                                    comply_cleverhans=False)

# spatial constrain
# DDN
DDN_N = DDNL2Attack(target_model, nb_iter=100, gamma=0.05, init_norm=1.0, quantize=True, levels=256, clip_min=0.0,
                        clip_max=1.0, targeted=False, loss_fn=None)


# STA
STA_N = SpatialTransformAttack(
    target_model, 10, clip_min=0.0, clip_max=1.0, max_iterations=5000, search_steps=20, targeted=False)





'''
PSSR
'''
from models.pix2pix_unet_dropout_2 import define_G1, define_G2, define_D, GANLoss, get_scheduler, update_learning_rate, define_G
from torch.autograd import Variable

def compute_kl_loss(p, q, pad_mask=None):  # 同一个样本的两次logits计算kl散度
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()
    loss = (p_loss + q_loss) / 2
    return p_loss


criterionGAN = GANLoss().cuda()
criterionL1 = nn.L1Loss().cuda()
criterionMSE = nn.MSELoss().cuda()
criterion_cls = nn.CrossEntropyLoss()

input_nc = 3
output_nc = 3
ngf = 64
ndf = 64

net_g = define_G(input_nc, output_nc, ngf, 'batch', False, 'normal', 0.02, gpu_id='cuda')
net_d = define_D(input_nc + output_nc, ndf, 'basic', gpu_id='cuda')




# setup optimizer
optimizer_g = optim.Adam(net_g.parameters(), lr=0.001, betas=(0.5, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=0.001, betas=(0.5, 0.999))
optimizer_target = torch.optim.Adam(target_model.parameters(), lr=0.001, betas=(0.9, 0.99))

epoch_count = 1
niter = 100
niter_decay = 100
lamb = 10
targetlabel = torch.zeros(BATCH_SIZE).cuda()
targetlabel = targetlabel.to('cuda', dtype=torch.int64)
target_model = torch.load('/media/cqu/D/FXV/PSSR_master/saving_models/CIFAR10/VGG16_CIFAR10.pkl').eval()

for epoch in range(epoch_count, 30):
    for iteration, batch in enumerate(trainloader, 1):
        imgs, labels = batch[0].cuda(), batch[1].cuda()
        targetlabel_temp = targetlabel[0:imgs.shape[0]]
        # if epoch == 1:
        imgs_pgdn = PGD_N(imgs)
        imgs_pgdt = PGD_T(imgs, targetlabel_temp)
        imgs_fgsmn = FGSM_N(imgs)
        imgs_fgsmt = FGSM_T(imgs, targetlabel_temp)
        #     torch.save(imgs_pgdn, './MNIST_ADV/pgdn03/' + str(iteration) +'.pt')
        #     torch.save(imgs_pgdt, './MNIST_ADV/pgdt03/' + str(iteration) + '.pt')
        #     torch.save(imgs_fgsmn, './MNIST_ADV/fgsmn03/' + str(iteration) + '.pt')
        #     torch.save(imgs_fgsmt, './MNIST_ADV/fgsmt03/' + str(iteration) + '.pt')
        # else:
        # imgs_pgdn = torch.load('./MNIST_ADV/pgdn03/' + str(iteration) + '.pt').cuda()
        # imgs_pgdt = torch.load('./MNIST_ADV/pgdt03/' + str(iteration) + '.pt').cuda()
        # imgs_fgsmn = torch.load('./MNIST_ADV/fgsmn03/' + str(iteration) + '.pt').cuda()
        # imgs_fgsmt = torch.load('./MNIST_ADV/fgsmt03/' + str(iteration) + '.pt').cuda()

        labels = torch.cat((labels, labels, labels, labels, labels), 0)
        real_a, real_b = torch.cat((imgs, imgs_fgsmn, imgs_fgsmt, imgs_pgdn, imgs_pgdt), 0), torch.cat((imgs, imgs, imgs, imgs, imgs), 0)  # a->b的转换

        # vutils.save_image(real_a, './saving_samples/MNIST/MAE_MNIST_saliency_p2p_rdrop/img_mix_{}.png'.format(epoch))
        # vutils.save_image(real_b, './saving_samples/MNIST/MAE_MNIST_saliency_p2p_rdrop/img_benign_{}.png'.format(epoch))
        # (0) Erase the real_a with saliency map
        ######################
        real_a = Variable(real_a, requires_grad=True)
        optimizer_target.zero_grad()
        predict = target_model(real_a)
        loss = criterion_cls(predict, labels)
        loss.backward()
        saliency = torch.abs(real_a.grad)  # 显著性图
        # vutils.save_image(saliency, './saving_samples/CIFAR10/MAE_CIFAR10_saliency_p2p_rdrop/saliency_map_{}.png'.format(epoch))
        zero_matrix = torch.zeros_like(saliency)
        one_matrix = torch.ones_like(saliency)
        mean_value = (saliency).sum().sum().sum() / (BATCH_SIZE * 32 * 32 * 1)
        mask = torch.where(saliency < mean_value * 1.5, one_matrix, zero_matrix)  # 显著性小于均值的才留为1（意思是保留），大于均值的就取0（意思是擦掉）
        real_a = real_a * mask
        # mask0 = torch.where(saliency < mean_value * 0.3, one_matrix, zero_matrix)
        # mask1 = torch.where(saliency < mean_value * 0.5, one_matrix, zero_matrix)  # 显著性小于均值的才留为1（意思是保留），大于均值的就取0（意思是擦掉）
        # mask2 = torch.where(saliency < mean_value * 0.7, one_matrix, zero_matrix)
        # mask3 = torch.where(saliency < mean_value * 0.9, one_matrix, zero_matrix)
        # mask4 = torch.where(saliency < mean_value * 1.3, one_matrix, zero_matrix)
        # mask5 = torch.where(saliency < mean_value * 1.7, one_matrix, zero_matrix)
        # mask6 = torch.where(saliency < mean_value * 2.0, one_matrix, zero_matrix)
        # mask7 = torch.where(saliency < mean_value * 1.5, one_matrix, zero_matrix)



        # vutils.save_image(mask, './saving_samples/CIFAR10/MAE_CIFAR10_saliency_p2p_rdrop/mask_{}.png'.format(epoch))
        # real_a0 = real_a * mask0
        # real_a1 = real_a * mask1
        # real_a2 = real_a * mask2
        # real_a3 = real_a * mask3
        # real_a4 = real_a * mask4
        # real_a5 = real_a * mask5
        # real_a6 = real_a * mask6
        # real_a7 = real_a * mask7
        # if iteration == 1:
        #     vutils.save_image(real_a[6], '/media/cqu/D/FXV/PSSR_master/reala6.png')
        #     vutils.save_image(mask0[6], '/media/cqu/D/FXV/PSSR_master/mask0.png')
        #     vutils.save_image(real_a0[6], '/media/cqu/D/FXV/PSSR_master/real0.png')
        #
        #     vutils.save_image(mask1[6], '/media/cqu/D/FXV/PSSR_master/mask1.png')
        #     vutils.save_image(real_a1[6], '/media/cqu/D/FXV/PSSR_master/real1.png')
        #
        #     vutils.save_image(mask2[6], '/media/cqu/D/FXV/PSSR_master/mask2.png')
        #     vutils.save_image(real_a2[6], '/media/cqu/D/FXV/PSSR_master/real2.png')
        #
        #     vutils.save_image(mask3[6], '/media/cqu/D/FXV/PSSR_master/mask3.png')
        #     vutils.save_image(real_a3[6], '/media/cqu/D/FXV/PSSR_master/real3.png')
        #
        #     vutils.save_image(mask4[6], '/media/cqu/D/FXV/PSSR_master/mask4.png')
        #     vutils.save_image(real_a4[6], '/media/cqu/D/FXV/PSSR_master/real4.png')
        #
        #     vutils.save_image(mask5[6], '/media/cqu/D/FXV/PSSR_master/mask5.png')
        #     vutils.save_image(real_a5[6], '/media/cqu/D/FXV/PSSR_master/real5.png')
        #
        #     vutils.save_image(mask6[6], '/media/cqu/D/FXV/PSSR_master/mask6.png')
        #     vutils.save_image(real_a6[6], '/media/cqu/D/FXV/PSSR_master/real6.png')
        #
        #     vutils.save_image(mask7[6], '/media/cqu/D/FXV/PSSR_master/mask7.png')
        #     vutils.save_image(real_a7[6], '/media/cqu/D/FXV/PSSR_master/real7.png')
        #     print(111111)
        ######################
        # (1) Update D network
        ######################
        fake_b = net_g(real_a)
        fake_b_2 = net_g(real_a)

        # fake_b_at_feat = net_g1(real_a[imgs.size(0):])
        # fake_b_st_feat = model_st(real_a[:imgs.size(0)])
        # fake_b_at = net_g2(fake_b_at_feat)
        # fake_b_st = net_g2(fake_b_st_feat)

        # if epoch < 5:
        #     fake_b_st_feat = model_st(real_a[:imgs.size(0)])
        # else:
        #     fake_b_st_feat = net_g1(real_a[:imgs.size(0)])



        # fake_b_2_at_feat = net_g1(real_a[imgs.size(0):])
        # fake_b_2_at = net_g2(fake_b_2_at_feat)
        # fake_b_2_st_feat = model_st(real_a[:imgs.size(0)])
        # fake_b_2_st = net_g2(fake_b_2_st_feat)


        # if epoch < 5:
        #     fake_b_2_st_feat = model_st(real_a[:imgs.size(0)])
        # else:
        #     fake_b_2_st_feat = net_g1(real_a[:imgs.size(0)])

        #
        # fake_b = torch.cat((fake_b_at, fake_b_st), 0)
        # fake_b_2 = torch.cat((fake_b_2_at, fake_b_2_st), 0)

        optimizer_d.zero_grad()

        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)  # 原图和生成图片拼接
        pred_fake = net_d.forward(fake_ab.detach())  # 然后一起预测
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = net_d.forward(real_ab)
        loss_d_real = criterionGAN(input=pred_real, target_is_real=True)

        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5
        loss_d.backward(retain_graph=True)
        optimizer_d.step()

        ######################
        # (2) Update G network
        ######################
        optimizer_g.zero_grad()

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        fake_ab_2 = torch.cat((real_a, fake_b_2), 1)
        pred_fake_2 = net_d.forward(fake_ab_2)
        loss_g_gan_2 = criterionGAN(pred_fake_2, True)

        # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * lamb
        loss_g_l1_2 = criterionL1(fake_b_2, real_b) * lamb

        # Third, query the classifier
        rec_query_1 = target_model(fake_b)
        loss_query = criterion_cls(rec_query_1, labels)
        rec_query_2 = target_model(fake_b_2)
        loss_query_2 = criterion_cls(rec_query_2, labels)
        real_query = target_model(real_b)

        # Forth, compute kl loss
        kl_loss = compute_kl_loss(rec_query_1, real_query) + compute_kl_loss(rec_query_2, real_query)

        loss_g = 1.0*(loss_query + loss_query_2) + 1*(loss_g_gan + loss_g_gan_2) + 1.0*(loss_g_l1 + loss_g_l1_2) + 0.5 * kl_loss
        #loss_g = 0.07 * kl_loss

        loss_g.backward()
        optimizer_g.step()


        if iteration % 200 == 0:
            print("===> Epoch[{}]({}/{}), Loss_D: {:.4f} Loss_G: {:.4f}".format(epoch, iteration, len(trainloader), loss_d.item(), loss_g.item()))

    if epoch % 2 == 0:
        torch.save(net_g, './saving_models/CIFAR10/MAE_927'+str(epoch)+'.pkl')

denoiser1 = torch.load('/media/cqu/D/FXV/PSSR_master/saving_models/CIFAR10/MAE_92718.pkl').eval()
total = 0
correct = 0
temp = 0
targetlabel = torch.zeros(BATCH_SIZE).cuda()
targetlabel = targetlabel.to('cuda', dtype=torch.int64)
for img, label in testloader:
    targetlabel_temp = targetlabel[0:img.shape[0]]
    img, label = img.cuda(), label.cuda()

    # 1. 获取对抗样本
    # img_adv = img
    img_adv = PGD_N(img)
    # img_adv = BIM_N.perturb(img)
    # img_adv = AA_N.run_standard_evaluation(img, label, bs=BATCH_SIZE)

    img_rec = denoiser1(img_adv)
    x = target_model(img_rec)
    _, prediction = torch.max(x, 1)
    total += label.size(0)
    correct += (prediction == label).sum()
    temp += 1
print('There are ' + str(correct.item()) + ' correct pictures.')
print('Accuracy=%.2f' % (100.00 * correct.item() / total))










