# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable

import numpy as np
import pywt
import scipy
import torch

import utils.misc as misc
import utils.lr_sched as lr_sched

from torchmetrics.classification import MultilabelAUROC
from timm.utils import accuracy
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.io import loadmat, savemat
import scipy.signal as signal
from ecg_preprocessing import Rwave_Detection,HF_Noise_Removal
from scipy.signal import find_peaks, find_peaks_cwt
import torch.nn.functional as F
from utils.loss import FocalLoss
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
# Load NeuroKit and other useful packages
import neurokit2 as nk
import pandas as pd
import scipy.stats as stats

num_categories = 3
nh=0
# noise_criterion = FocalLoss(gamma=2, weight=None)
# noise_criterion = torch.nn.CrossEntropyLoss()
noise_criterion = torch.nn.MSELoss()
# noise_criterion = torch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean', beta=1.0)
# denoise_criterion = torch.nn.MSELoss()
# SmoothL1 = torch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean', beta=1.0)

class MultiTaskLossWrapper(torch.nn.Module):
    """ Wraps multiple losses with learnable uncertainty weighting """
    def __init__(self, num_tasks):
        super(MultiTaskLossWrapper, self).__init__()
        self.log_vars = torch.nn.Parameter(torch.zeros(num_tasks))  # log(sigma^2)

    def forward(self, losses):
        weighted_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss += precision * loss + self.log_vars[i]
            # print("precision", precision)
        return weighted_loss

import torch
import torch.nn as nn

class GradNormLossWrapper(nn.Module):
    def __init__(self, num_tasks, record_task_weights=False, log_writer=None):
        super().__init__()
        self.num_tasks = num_tasks
        # 初始化 task weights 为 1.0，且是可学习参数
        self.task_weights = nn.Parameter(torch.ones(num_tasks, dtype=torch.float32))
        self.record_task_weights = record_task_weights
        self.log_writer = log_writer
        self.step = 0  # 用来在tensorboard里记录

    def forward(self, losses, shared_parameters):
        """
        losses: list of task losses
        shared_parameters: list of model parameters that are shared
        """
        assert len(losses) == self.num_tasks, "Mismatch between number of tasks and provided losses."

        # 计算每个loss的梯度
        grad_norms = []
        for loss in losses:
            grad = torch.autograd.grad(
                loss, shared_parameters,
                retain_graph=True,
                create_graph=True,
                allow_unused=True
            )
            # 有些 grad 是 None，替换成 zeros
            grads = [(g if g is not None else torch.zeros_like(p)) for g, p in zip(grad, shared_parameters)]
            # 计算整体 norm
            grad_norm = torch.norm(torch.stack([g.norm() for g in grads]))
            grad_norms.append(grad_norm)

        # 计算加权 loss
        weighted_losses = [w * l for w, l in zip(self.task_weights, losses)]
        total_loss = sum(weighted_losses)

        # 平衡 loss 的 scale
        # normalize each task's grad norm 和 total grad norm，使优化平滑
        avg_grad_norm = sum(grad_norms) / self.num_tasks
        loss_balance = 0
        for i in range(self.num_tasks):
            loss_balance += (grad_norms[i] - avg_grad_norm).abs()

        total_loss += 0.1 * loss_balance  # 0.1是一个可调的超参数，用来balance

        # 记录每步的 task weights
        if self.record_task_weights and self.log_writer is not None:
            for i in range(self.num_tasks):
                self.log_writer.add_scalar(f'task_weight/task_{i}', self.task_weights[i].item(), self.step)
            self.step += 1

        return total_loss


def denoise_criterion(target, pred, label):
    label = torch.where(label == 0, torch.tensor(1), label)
    label = torch.where(label == 2, torch.tensor(0), label)
    # # SmoothL1
    # loss = 0
    # count = 0
    # for i in range(target.size(0)):
    #     if label[i] != 0:
    #         c=SmoothL1(target[i,:], pred[i,:])
    #         loss += c
    #         count += 1
    # loss = loss/count
    device = torch.device('cuda')
    # MSE
    # pred = torch.diff(pred, dim=-1)
    # a=torch.zeros(pred.size()[0], 4).to(device)
    # pred1 = torch.cat((a,pred), dim=1)
    # pred = apply_gaussian_smoothing(pred, sigma=1.0).to(device)
    # target1 = apply_gaussian_smoothing(target, sigma=1.0)

    loss = abs(pred - target) ** 2
    diff_loss = torch.diff(pred, dim=-1)
    # u = torch.mean(pred, dim=-1)
    # sigma = torch.std(pred, dim=-1)
    '''
    for i in range(loss.size(0)):
        # dict = {}
        # dict['ecg'] = target[i].detach().cpu().numpy()
        # dict['loss'] = loss[i].detach().cpu().numpy()
        if label[i] == 0:
            ind,a = wavelet_dec(target[i].detach().cpu().numpy())
            # dict['pos'] = a
            a = torch.tensor(a).to(device)
            loss[i, :] = torch.where(a>0, abs(2) * loss[i], loss[i])
    '''
        # if label[i] == 1:
        #     flag = QRS_Position(target[i].detach().cpu().numpy(),200)
        #     # dict['pos'] = flag
        #     flag = torch.tensor(flag).to(device)
        #     loss[i, :] = torch.where(flag>0, abs(3) * loss[i], loss[i])
        # dict['w_loss'] = loss[i].detach().cpu().numpy()
        # savemat('loss.mat', {'dict': dict})
        # if label[i] == 1:
        #     ind = wavelet_dec(target[i].detach().cpu().numpy())
        #     loss[i, :] = torch.where(ind, abs(2) * loss[i], loss[i])
        #     dict = {}
        #     dict['ecg'] = target[i].detach().cpu().numpy()
        #     dict['loss'] = loss[i].detach().cpu().numpy()
        #     pos = (target[i] >= u[i] + 2*sigma[i])
        #     loss[i,:] = torch.where(pos, abs(2) * loss[i], loss[i])
        #     dict['w_loss'] = loss[i].detach().cpu().numpy()
        # if label[i] == 0:
        #     pos = (target[i] <= u[i] + sigma[i]) & (target[i] >= u[i] - sigma[i])
        #     loss[i,:] = torch.where(pos, abs(2) * loss[i], loss[i])
        #     dict['w_loss'] = loss[i].detach().cpu().numpy()
    #         savemat('loss.mat', {'dict': dict})
    loss = loss.mean(dim=-1)
    loss = (loss * label).sum() / label.sum()
    diff_loss = diff_loss.mean(dim=-1)
    diff_loss = (diff_loss * label).sum() / label.sum()
    loss1 = loss + 0.5 * diff_loss


    return loss1
def Baseline_Removal(raw_ecg1, fs):
	baseline1 = signal.medfilt(raw_ecg1.squeeze(), round(0.2 * fs) + 1)
	baseline2 = signal.medfilt(raw_ecg1.squeeze(), round(0.6 * fs) + 1)
	baseline = 0.5 * baseline1 + 0.5 * baseline2
	ecg_baseline_eliminated = raw_ecg1 - baseline
    # savemat('baseline.mat', {'baseline1': baseline1,'baseline2': baseline2})
	return ecg_baseline_eliminated

def scaled_coefficient(clean_ecg_energy, noise, target_snr):
    adjusted_noise_energy = clean_ecg_energy / (10 ** (target_snr / 10))
    noise_energy = np.sum(noise ** 2)
    coefficient = np.sqrt(adjusted_noise_energy / noise_energy)
    return coefficient

def find_fidualpoint(signal, r_pos):
    peaks_list = []
    for j in range(len(r_pos)-1):
        tmp = signal[r_pos[j] : r_pos[j+1]]
        peaks = find_peaks_cwt(tmp, np.arange(1, 200*0.3))
        # savemat('peaks.mat', {'peaks': peaks,
        #                     'ecg':signal})
        peaks_list.append(1) if len(peaks) > 0 else 0
    if len(peaks_list) >= 0.5:
        return 1, peaks_list
    else:
        return 0, np.array(peaks_list)

def patchify(imgs):
    pw = 50
    w = imgs.shape[2] // pw
    x = imgs.reshape((imgs.shape[0], w, pw))
    return x

def unpatchify(x):
    ph = 1
    pw = 50
    h = 1
    w = x.shape[1] // 1
    imgs = x.reshape((x.shape[0],  h * ph, w * pw))
    return imgs

def random_masking(x):
    x_masked = patchify(x.copy())
    mask_ratio = np.random.rand()
    N, L, D = x_masked.shape  # batch, length, dim
    len_keep = int(L * (1-mask_ratio))
    noise = np.random.rand(N, L)
    ids_shuffle = np.argsort(noise, axis=1)
    ids_mask = ids_shuffle[:, len_keep+1:]
    x_masked[:, ids_mask, :] = 0
    x_masked = unpatchify(x_masked)
    # savemat('mask.mat', {'mask': x_masked.squeeze(),
    #                      'raw': x.squeeze()})

    return x_masked

def wavelet_dec(signal, wavelet='db6', level=4):
    # coeffs = pywt.wavedec(signal, wavelet, mode="per", level=level)
    # cA4 = coeffs[0]
    # app_coeffs = [cA4, None] + [None] * 4
    # reconstructed_signal = pywt.waverec([app_coeffs], wavelet)

    ca = []  # 近似分量
    cd = []  # 细节分量
    a = signal.copy()
    for i in range(level):
        (a, d) = pywt.dwt(a, wavelet)  # 进行6阶离散小波变换
        ca.append(a)
        cd.append(d)
    rec_a = []  # 近似系数重建后的时域波形
    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, wavelet))  # 近似系数重构
    # savemat('baseline.mat', {'rec_a': rec_a, 'signal': signal})

    ind = np.where(rec_a[-1][:-2] >= 0)[0]

    return ind,rec_a[-1][:-2]

def QRS_Position(signal, fs):
    r_pos = Rwave_Detection(signal, fs)
    flag = np.zeros(signal.shape)
    for i in range(1, len(r_pos)-1):
        flag[r_pos[i] - 15: r_pos[i] + 15] = 1
    if r_pos[0] >= 15:
        flag[r_pos[0] - 15 : r_pos[0] + 15] = 1
    else:
        flag[0 : r_pos[0] + 15] = 1
    if r_pos[-1] <= 999-15:
        flag[r_pos[-1] - 15 : r_pos[-1] + 15] = 1
    else:
        flag[r_pos[-1] - 15 : 999] = 1
    savemat('qrs.mat', {'r_pos': r_pos, 'flag':flag, 'ECG':signal})
    return flag
def gaussian_kernel(sigma=1.0, kernel_size=5):
    k = kernel_size // 2
    x = np.arange(-k, k + 1)
    g = np.exp(-0.5 * (x / sigma) ** 2)
    g = g / g.sum()
    return g

def apply_gaussian_smoothing(image, sigma=1.0):
    kernel = gaussian_kernel(sigma).reshape(1,1,-1) # 一维卷积核
    # kernel = kernel.repeat(64, 0) # 一维卷积核
    # kernel = kernel.repeat(64, 1)
    # i = image.unsqueeze(0).unsqueeze(0)
    # k = torch.tensor(kernel)
    x = image.detach().cpu().unsqueeze(1)
    smoothed_image = F.conv1d(x, torch.tensor(kernel,dtype=torch.float16), padding=2)
    return smoothed_image.squeeze(1)

def wavelet_denoising(signal, wavelet_name='db6', level=5, threshold=0.1):
    w = pywt.Wavelet(wavelet_name)
    maxlev = pywt.dwt_max_level(len(signal), w.dec_len)
    coeffs = pywt.wavedec(signal, wavelet_name, level=maxlev)
    thresholded_coeffs = [pywt.threshold(c, value=threshold*max(c)) for c in coeffs]
    # thresholded_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    denoised_signal = pywt.waverec(thresholded_coeffs, wavelet_name)
    return denoised_signal

def add_various_noise(signal, status):
    # signal = np.random.normal(0, 1, (64, 1, 1, 1000))
    global nh
    status = 'train'
    assert status in ['train', 'test', 'val']
    type_map = {'train': 2880, 'val': 360, 'test': 362}
    len = type_map[status]
    signal = signal.detach().numpy()
    signal_energy = np.sum(signal ** 2, axis=-1).squeeze()
    # gaussian_noise = np.random.normal(0, 1, signal.shape)
    # noise_energy = np.sum(gaussian_noise ** 2, axis=-1).squeeze()
    label = np.random.randint(0, 3, size=signal.shape[0])
    noisy_signal = np.zeros(signal.shape)
    noisy_label = np.zeros(signal.shape)
    random_snr = np.zeros(label.shape)

    for i, l in enumerate(label):
        # if i ==44 :
        #     a=1
        # print(i)
        r_pos = Rwave_Detection(signal[i].squeeze(), 200)
        diff = np.diff(r_pos)
        r_ind = np.where(diff > 10)[0]
        r_pos = r_pos[r_ind]
        r_pos = np.insert(r_pos, 0, 0)
        r_pos = np.append(r_pos, 999)
        r_pos = np.unique(r_pos)
        RRI_en = np.zeros(r_pos.shape[0]-1)
        mixed = np.random.randint(1,4)
        extra_noise = 0
        if l == 0:
            random_snr[i] = np.random.randint(16, 30)
            # signal[i]=  wavelet_denoising(signal[i].squeeze(), 'db8', 4, 0.1)
            # signal[i] = HF_Noise_Removal(signal[i].squeeze(), 200, 30)
            # noisy_label[i] = np.ones(signal[i].shape)
            noisy_label[i] = np.zeros(signal[i].shape)
        elif l == 1:
            random_snr[i] = np.random.randint(5, 14)
            # rnd1 = np.random.rand()
            # if rnd1 <= 0.1:
            #     extra_noise = 1
            #     mixed = 0
            #     random_snr[i] = np.random.randint(2, 9)
            #     gaussian_noise = np.random.normal(0, 1, signal[i].shape)
            #     coefficient = scaled_coefficient(signal_energy[i], gaussian_noise, random_snr[i])
            #     noisy_signal[i] = gaussian_noise* coefficient + signal[i]
            #     noisy_label[i] = np.ones(signal[i].shape)
            #     # fig, ax = plt.subplots(2,1,figsize=(25, 5))
            #     # ax[0].plot(noisy_signal[i].squeeze())
            #     # ax[1].plot(signal[i].squeeze())
            #     # plt.savefig(str(i) + '_' + str(random_snr[i]) + '_' + str(label[i]) + '.jpg')
            #     continue
            # signal[i]=  wavelet_denoising(signal[i].squeeze(), 'db8', 4, 0.1)
            # signal[i] = HF_Noise_Removal(signal[i].squeeze(), 200, 30)
            # noisy_label[i] = np.zeros(signal[i].shape)
            noisy_label[i] = np.ones(signal[i].shape)
        elif l == 2:
            rnd = np.random.rand()
            if rnd < 0.1:
                extra_noise = 1
                mixed = 0
                name = str(np.random.randint(1, 872+1)) + '.mat'
                path = '.\\data_h5\\noise_Data_ex\\' + name
                noise = loadmat(path)['tmp1']
                # noise = np.transpose(noise)
                noisy_signal[i] = noise
                # noisy_label[i] = np.zeros(signal[i].shape)
                noisy_label[i] = np.ones(signal[i].shape)
                continue
            # if rnd > 0.95:
            #     extra_noise = 1
            #     mixed = 0
            #     name = str(np.random.randint(1, 500 + 1)) + '.mat'
            #     path = '.\\data_h5\\LAB_BAD\\' + name
            #     noise = loadmat(path)['tmp1']
            #     # noise = np.transpose(noise)
            #     noisy_signal[i] = noise
            #     # noisy_label[i] = np.zeros(signal[i].shape)
            #     noisy_label[i] = np.ones(signal[i].shape)
            #     continue
            random_snr[i] = np.random.randint(-12, 3)
            # noisy_label[i] = np.zeros(signal[i].shape)
            noisy_label[i] = np.ones(signal[i].shape)
        if mixed == 1:
            name = 'em'  + str(np.random.randint(1, len+1)) + '.mat'
            path = '.\\data_h5\\noise_Data_' + str(status) + '\\' + name
            em = loadmat(path)['eletrode_artifacts']; em = np.transpose(em)
            em_ = Baseline_Removal(em,200)
            coefficient = scaled_coefficient(signal_energy[i], em_, random_snr[i])
            noisy_signal[i] = signal[i] + coefficient * em
            n = em_
            noisy_bw = signal[i] + coefficient * em_
        elif mixed == 2:
            name = 'ma'  + str(np.random.randint(1, len+1)) + '.mat'
            path = '.\\data_h5\\noise_Data_' + str(status) + '\\' + name
            ma = loadmat(path)['EMG']; ma = np.transpose(ma)
            ma_ = Baseline_Removal(ma, 200)
            coefficient = scaled_coefficient(signal_energy[i], ma_, random_snr[i])
            noisy_signal[i] = signal[i] + coefficient * ma
            noisy_bw = signal[i] + coefficient * ma_
            n = ma_
        elif mixed == 3:
            # if extra_noise == 1:
            #     continue
            name = 'em'  + str(np.random.randint(1, len+1)) + '.mat'
            path = '.\\data_h5\\noise_Data_' + str(status) + '\\' + name
            em = loadmat(path)['eletrode_artifacts']; em = np.transpose(em)
            name = 'ma'  + str(np.random.randint(1, len+1)) + '.mat'
            path = '.\\data_h5\\noise_Data_' + str(status) + '\\' + name
            ma = loadmat(path)['EMG']; ma = np.transpose(ma)
            em_ma_ = Baseline_Removal(em+ma, 200)
            coefficient = scaled_coefficient(signal_energy[i], em_ma_, random_snr[i])
            noisy_signal[i] = signal[i] + coefficient * ma + coefficient * em
            noisy_bw = signal[i] + coefficient * em_ma_
            n = em_ma_

        if extra_noise == 0:
            n_en = np.zeros(r_pos.shape[0]-1)
            r_l = np.zeros(r_pos.shape[0]-1)
            mse_l = np.zeros(r_pos.shape[0]-1)
            # mae = np.zeros(r_pos.shape[0]-1)
            sig_LP = wavelet_denoising(noisy_bw.squeeze(), 'db8', 4, 0.1)
            # sig_LP = HF_Noise_Removal(sig_LP.squeeze(), 200, 30)
            # sig_LP = scipy.signal.medfilt(noisy_bw.squeeze(), 5)
            for k in range(r_pos.shape[0]-1):
                # cs[k] = cosine_similarity(sig_LP[r_pos[k]:r_pos[k+1]].reshape(1, -1), signal[i][0][0][r_pos[k]:r_pos[k+1]].reshape(1, -1) ** 2)[0][0]
                RRI_en[k] = np.sum(signal[i][0][0][r_pos[k]:r_pos[k+1]] ** 2)
                n_en[k] = np.sum((sig_LP-signal[i][0][0])[r_pos[k]:r_pos[k+1]] ** 2)
                r_l[k], _ = stats.pearsonr(signal[i][0][0][r_pos[k]:r_pos[k+1]], sig_LP[r_pos[k]:r_pos[k+1]])
                mse_l[k] = np.mean((sig_LP-signal[i][0][0])[r_pos[k]:r_pos[k+1]]**2)
                # mae[k] = np.max(abs(sig_LP-signal[i][0][0])[r_pos[k]:r_pos[k+1]])
                # n_en[k] = np.sum(coefficient*n[0][r_pos[k]:r_pos[k+1]] ** 2)
            eps = np.finfo(signal.dtype).eps
            n_en = np.maximum(n_en, eps)
            snr_n = 10 * np.log10(RRI_en / n_en+1e-5)
            # savemat('raw.mat',{'raw': signal[i].squeeze(),
            #                    'noisy': noisy_signal[i].squeeze(),
            #                    'noise': em.squeeze(),
            #                    'noise_no_bw': em_.squeeze(),
            #                    'noise_label': noisy_label[i].squeeze(),
            #                    'snr': random_snr[i]})
            # del em, em_, em_en, snr_em
            rri_count = 0
            for k in range(snr_n.shape[0]):
                if snr_n[k]>=9 or r_l[k]>0.905 or mse_l[k]<=0.21:
                    noisy_label[i][0][0][r_pos[k]:r_pos[k+1]+1] = 0
                if snr_n[k] >= -0.5:
                    rri_count += 1

            # if np.sum(noisy_label[i])==0 and l==1:
            #     label[i] = 0
            #     # print(str(i)+"A")
            # if np.sum(noisy_label[i])==0 and l==2:
            #     label[i] = 1
            #     # print(str(i)+"B")
            # if rri_count > snr_n.shape[0]/2 and l==2 and random_snr[i]>-3:
            #     label[i] = 1
            #     print(str(i)+"C")

        '''
        time_axis = np.arange(0, 1000)
        fig, axes = plt.subplots(3, 1, figsize=(25, 15), sharex=True, gridspec_kw={'wspace': 0.2, 'hspace': 0.5})
        axes[0].plot(time_axis, signal[i].squeeze(), alpha=0.9, color='red')
        axes[1].plot(np.diff(signal[i].squeeze()), alpha=0.9, color='red')
        axes[2].plot(np.diff(signal[i].squeeze(), n=2), alpha=0.9, color='red')
        if extra_noise == 0:
            # axes[0].plot(time_axis, sig_LP.squeeze(), alpha=0.9)
            axes[0].plot(noisy_signal[i].squeeze(), alpha=0.9)
            axes[1].plot(np.diff(sig_LP.squeeze()), alpha=0.9)
            axes[2].plot(np.diff(sig_LP.squeeze(), n=2), alpha=0.9)
            axes[1].set_title(str(np.round(snr_n, 1))+'/n'+str(np.round(mse_l, 2)), fontdict={'family': 'serif', 'weight': 'bold', 'size': 30})
            axes[2].set_title(str(np.round(r_l, 3) ), fontdict={'family': 'serif', 'weight': 'bold', 'size': 30})
            # mae1 = np.max(abs(np.diff(sig_LP.squeeze()) - np.diff(signal[i].squeeze())))
            # mae2 = np.max(abs(np.diff(sig_LP.squeeze(),n=2) - np.diff(signal[i].squeeze(),n=2)))
            # # axes[0].plot(time_axis, noisy_signal[i].squeeze(), alpha=0.7)

            # axes[0].plot(time_axis, noisy_bw.squeeze(), alpha=0.7)
            # axes[1].plot(np.diff(noisy_bw.squeeze()), alpha=0.9)
            # axes[2].plot(np.diff(noisy_bw.squeeze(), n=2), alpha=0.9)
            # mae1 = np.max(abs(np.diff(noisy_bw.squeeze()) - np.diff(signal[i].squeeze())))
            # mae2 = np.max(abs(np.diff(noisy_bw.squeeze(),n=2) - np.diff(signal[i].squeeze(),n=2)))

            # axes[1].plot(time_axis, signal[i].squeeze(), alpha=0.7, color='red')
            # axes[1].plot(time_axis, sig_LP.squeeze(), alpha=0.9)
            # # axes[1].plot(time_axis, noisy_bw.squeeze(), alpha=0.7, color='red')
            # axes[0].set_title(str(np.round(snr_n, 1)), fontdict={'family': 'serif', 'weight': 'bold', 'size': 30})
        axes[0].plot(time_axis, (noisy_label[i] - 2).squeeze(), linewidth=1, alpha=0.9, color='green')
        axes[0].set_title(str(label[i]), fontdict={'family': 'serif', 'weight': 'bold', 'size': 30})
        # axes[1].set_title(str(mae1), fontdict={'family': 'serif', 'weight': 'bold', 'size': 30})
        # axes[2].set_title(str(mae2), fontdict={'family': 'serif', 'weight': 'bold', 'size': 30})

            # # sig_g= apply_gaussian_smoothing(torch.tensor(signal[i][0][0].copy()), 1).detach().numpy()
            # # axes[2].plot(time_axis, signal[i].squeeze(), alpha=0.7, color='red')
            # # axes[2].plot(sig_g.squeeze(), alpha=0.9)
        plt.savefig(str(i) + '_' + str(random_snr[i]) + '_' + str(label[i]) + '.jpg')
        plt.close()
        nh+=1
        # if i==15:
        #     break
    '''
    return torch.tensor(noisy_signal), label, random_snr, torch.tensor(noisy_label)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    multitask_loss_wrapper: torch.nn.Module,
                    log_writer=None,
                    args=None,):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    # shared_parameters = [p for p in model.parameters() if p.requires_grad]
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # if data_iter_step ==16:
        #     pause = 1
        # print(data_iter_step)
        noisy_samples, label, _ , noiseLabel= add_various_noise(samples, 'train')
        # noisy_samples = samples
        # label = np.array([0]*samples.shape[0])
        noisy_samples = noisy_samples.half()
        samples = samples.half()
        targets = torch.zeros(label.shape[0], num_categories, dtype=torch.float32)
        targets[np.arange(label.shape[0]), label.astype(int)] = 1
        label = torch.tensor(label)

        # clean =  samples[:,:,:, 1000:]
        # samples = samples[:,:,:, :1000]
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        if args.cuda is not None:
            with torch.amp.autocast('cuda'):
                # denoise_loss, _, outputs = model(samples.to(device), samples.to(device))
                # outputs, l, _, pred = model(noisy_samples.to(device), samples.to(device), label.to(device))
                outputs, l, _, pred = model(noisy_samples.to(device))

                # outputs = torch.sigmoid(outputs)
                # confidence_mask = (outputs > 0.9) | (outputs < 0.1)
                # corrected_labels = torch.where(confidence_mask, outputs, torch.tensor(targets).to(device))
                # class_loss = criterion(outputs, corrected_labels)
                # denoise_loss = denoise_criterion(samples.squeeze().to(device), pred.squeeze(), label.to(device))
                # noiseLabel = noiseLabel.squeeze().to(device).to(torch.float32)
                # l = torch.sigmoid(l.squeeze()).to(torch.float32)
                # confidence_mask_n = (l > 0.8) | (l < 0.1)
                # corrected_labels_n = torch.where(confidence_mask_n, l, noiseLabel)
                # noiseLabel_loss = noise_criterion(l, corrected_labels_n)

                class_loss = criterion(outputs, torch.tensor(targets).to(device))
                denoise_loss = denoise_criterion(samples.squeeze().to(device), pred.squeeze(), label.to(device))
                noiseLabel = noiseLabel.squeeze().to(device).to(torch.float32)
                l = torch.sigmoid(l.squeeze()).to(torch.float32)
                noiseLabel_loss = noise_criterion(l, noiseLabel)
                # noiseLabel_loss = noise_criterion(l, noiseLabel)/1000
                # noiseLabel_loss = noise_criterion(l.squeeze(), torch.sigmoid(noiseLabel.squeeze().to(device)))/1000
                # noiseLabel_loss = noiseLabel_loss.half()

        denoise_loss_value = denoise_loss.item()
        class_loss_value = class_loss.item()
        noiseLabel_loss_value = noiseLabel_loss.item()

        if not math.isfinite(class_loss_value):
            print("Class Loss is {}, stopping training".format(class_loss_value))
            sys.exit(1)

        # denoise_loss_value /= accum_iter
        # loss_scaler(class_loss + (noiseLabel_loss_value/class_loss_value) * noiseLabel_loss + (denoise_loss_value/class_loss_value) * denoise_loss, optimizer, parameters=model.parameters(),
        #             update_grad=(data_iter_step + 1) % accum_iter == 0)

        total_loss = multitask_loss_wrapper([class_loss, noiseLabel_loss, denoise_loss])
        # total_loss = multitask_loss_wrapper([class_loss, noiseLabel_loss, denoise_loss], shared_parameters)

        loss_scaler(total_loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        # loss_scaler(1*class_loss+1*noiseLabel_loss+2*denoise_loss, optimizer, parameters=model.parameters(),
        #             update_grad=(data_iter_step + 1) % accum_iter == 0)
        # loss_scaler(0*class_loss+0*noiseLabel_loss+1*denoise_loss, optimizer, parameters=model.parameters(),
        #             update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        if args.cuda is not None:
            torch.cuda.synchronize()


        metric_logger.update(denoise_loss=denoise_loss_value)
        metric_logger.update(class_loss=class_loss_value)
        metric_logger.update(noiseLabel_loss=noiseLabel_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        denoise_loss_value_reduce = misc.all_reduce_mean(denoise_loss_value)
        class_loss_value_reduce = misc.all_reduce_mean(class_loss_value)
        noiseLabel_loss_value_reduce = misc.all_reduce_mean(noiseLabel_loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss_denoise', denoise_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_class', class_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_noiseLabel', noiseLabel_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args):
    criterion_class = torch.nn.BCEWithLogitsLoss()
    # criterion_denoise = torch.nn.MSELoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    auc_meter = misc.SmoothedValue(window_size=1, fmt='{avg:.4f}')

    metric_logger.add_meter('auc', auc_meter)  # Add this line
    auc_meter.update(0)
    header = 'Val:'

    # switch to evaluation mode
    model.eval()
    trues = []
    preds = []

    for (samples, _) in metric_logger.log_every(data_loader, 10, header):
        noisy_samples, label, snr, noiseLabel = add_various_noise(samples, 'val')
        # noisy_samples = samples
        # label = np.array([0] * samples.shape[0])
        noisy_samples = noisy_samples.half()
        samples = samples.half()
        targets = torch.zeros(label.shape[0], num_categories, dtype=torch.float32)
        targets[np.arange(label.shape[0]), label.astype(int)] = 1
        label = torch.tensor(label)

        # if(args.classf_type != "multi_label"):
        #     target = target[:, 0]
        if args.cuda is not None:
            noisy_samples = noisy_samples.to(device, non_blocking=True)
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            noiseLabel = noiseLabel.to(device, non_blocking=True)

        # compute output
        if args.cuda is not None:
            with torch.amp.autocast('cuda'):
                # denoise_loss, _, outputs = model(samples, samples)
                # outputs, l, _, pred = model(noisy_samples, samples, label)
                outputs, l, _, pred = model(noisy_samples)
                denoise_loss = denoise_criterion(samples.squeeze(), pred.squeeze(), label)
                class_loss = criterion_class(outputs, targets)
                noiseLabel = noiseLabel.squeeze().to(device).to(torch.float32)
                l = torch.sigmoid(l.squeeze()).to(torch.float32)
                noiseLabel_loss = noise_criterion(l, noiseLabel)


        if(args.classf_type != "multi_label"):
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            batch_size = samples.shape[0]
            # metric_logger.update(class_loss=class_loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        else:
            acc1 = accuracy_score(targets.cpu(), torch.sigmoid(outputs.cpu()) >= 0.5)*100
            acc2 = accuracy_score(noiseLabel.cpu(), l.cpu() >= 0.5)*100
            # ml_auroc = MultilabelAUROC(num_labels=args.nb_classes, average="macro", thresholds=None)
            # auc =  roc_auc_score(targets.cpu().int(), torch.sigmoid(outputs.cpu()))
            batch_size = samples.shape[0]
            metric_logger.update(class_loss=class_loss.item())
            metric_logger.update(noiseLabel_loss=noiseLabel_loss.item())
            metric_logger.update(denoise_loss=denoise_loss.item())
            metric_logger.meters['acc1'].update(acc1, n=batch_size)
            metric_logger.meters['acc2'].update(acc2, n=batch_size)
            # metric_logger.meters['auc'].update(auc, n=batch_size)
            trues.append(targets.cpu().int())
            preds.append(torch.sigmoid(outputs.detach().cpu()))
    # gather the stats from all processes

    metric_logger.synchronize_between_processes()
    ml_auroc = MultilabelAUROC(num_labels=args.nb_classes, average="macro", thresholds=None)
    auc = ml_auroc(torch.cat(preds), torch.cat(trues))
    # auc =  roc_auc_score(torch.cat(trues), torch.cat(preds))
    metric_logger.meters['auc'].update(auc)  # Update the AUC meter

    print('* Acc@1 {top1.global_avg:.3f} auc {aucs:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, aucs = auc, losses=metric_logger.noiseLabel_loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test(data_loader, model, device, args, clean=False):
    criterion_class = torch.nn.BCEWithLogitsLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    auc_meter = misc.SmoothedValue(window_size=1, fmt='{avg:.4f}')

    metric_logger.add_meter('auc', auc_meter)  # Add this line
    auc_meter.update(0)
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    trues = []
    preds = []

    for (samples, _) in metric_logger.log_every(data_loader, 10, header):
        if clean:
            # score = np.array([15] * samples.shape[0])
            label = np.array([0] * samples.shape[0])
            noisy_samples = samples
        else:
            noisy_samples, label, snr, noiseLabel = add_various_noise(samples, 'test')
        # noisy_samples = samples
        # label = np.array([0] * samples.shape[0])
        noisy_samples = noisy_samples.half()
        samples = samples.half()
        targets = torch.zeros(label.shape[0], num_categories, dtype=torch.float32)
        targets[np.arange(label.shape[0]), label.astype(int)] = 1
        label = torch.tensor(label)

        # if(args.classf_type != "multi_label"):
        #     target = target[:, 0]
        if args.cuda is not None:
            noisy_samples = noisy_samples.to(device, non_blocking=True)
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            noiseLabel = noiseLabel.to(device, non_blocking=True)

        # compute output
        if args.cuda is not None:
            with torch.amp.autocast('cuda'):
                # denoise_loss, _, outputs = model(samples, samples)
                # outputs, l, _, pred = model(noisy_samples, samples, label)
                outputs, l, _, pred = model(noisy_samples)
                denoise_loss = denoise_criterion(samples.squeeze(), pred.squeeze(), label)
                class_loss = criterion_class(outputs, targets)
                noiseLabel = noiseLabel.squeeze().to(device).to(torch.float32)
                l = torch.sigmoid(l.squeeze()).to(torch.float32)
                noiseLabel_loss = noise_criterion(l, noiseLabel)

        if(args.classf_type != "multi_label"):
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            batch_size = samples.shape[0]
            # metric_logger.update(class_loss=class_loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        else:
            acc1 = accuracy_score(targets.cpu(), torch.sigmoid(outputs.cpu()) > 0.5)*100
            acc2 = accuracy_score(noiseLabel.cpu(), l.cpu() >= 0.5)*100
            # ml_auroc = MultilabelAUROC(num_labels=args.nb_classes, average="macro", thresholds=None)
            # auc = ml_auroc(torch.sigmoid(outputs.cpu()), targets.cpu().int())
            batch_size = samples.shape[0]
            metric_logger.update(class_loss=class_loss.item())
            metric_logger.update(noiseLabel_loss=noiseLabel_loss.item())
            metric_logger.update(denoise_loss=denoise_loss.item())
            metric_logger.meters['acc1'].update(acc1, n=batch_size)
            metric_logger.meters['acc2'].update(acc2, n=batch_size)
            # metric_logger.meters['auc'].update(auc, n=batch_size)
            trues.append(targets.cpu().int())
            preds.append(torch.sigmoid(outputs.detach().cpu()))
    # gather the stats from all processes

    # metric_logger.synchronize_between_processes()
    ml_auroc = MultilabelAUROC(num_labels=args.nb_classes, average="macro", thresholds=None)
    auc = ml_auroc(torch.cat(preds), torch.cat(trues))
    metric_logger.meters['auc'].update(auc)  # Update the AUC meter

    print('* Acc@1 {top1.global_avg:.3f} auc {aucs:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, aucs = auc, losses=metric_logger.noiseLabel_loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_lab(data_loader, model, device, args, clean=False):
    criterion_class = torch.nn.BCEWithLogitsLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    auc_meter = misc.SmoothedValue(window_size=1, fmt='{avg:.4f}')

    metric_logger.add_meter('auc', auc_meter)  # Add this line
    auc_meter.update(0)
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    trues = []
    preds = []

    for (samples, _) in metric_logger.log_every(data_loader, 10, header):

        target_ecg = samples[:, :, :, :-1]
        label = np.array(samples[:, :, :, -1]).squeeze(-1).squeeze(-1)
        targets = torch.zeros(label.shape[0], num_categories, dtype=torch.float32)
        targets[np.arange(label.shape[0]), label.astype(int)] = 1
        # label = label.squeeze(-1).squeeze(-1)
        label = torch.tensor(label)

        target_ecg = target_ecg.half()
        if args.cuda is not None:
            target_ecg = target_ecg.to(device, non_blocking=True)
            # targets = targets.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            # noiseLabel = noiseLabel.to(device, non_blocking=True)

        # compute output
        if args.cuda is not None:
            with torch.amp.autocast('cuda'):
                # outputs, l, _, pred = model(target_ecg, target_ecg, label)
                outputs, l, _, pred = model(target_ecg)


        if(args.classf_type != "multi_label"):
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            batch_size = samples.shape[0]
            # metric_logger.update(class_loss=class_loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        else:
            acc1 = accuracy_score(targets.cpu(), torch.sigmoid(outputs.cpu()) > 0.5)*100

            batch_size = target_ecg.shape[0]

            metric_logger.meters['acc1'].update(acc1, n=batch_size)

            # trues.append(targets.cpu().int())
            # preds.append(torch.sigmoid(outputs.detach().cpu()))
    # gather the stats from all processes

    # metric_logger.synchronize_between_processes()
    # ml_auroc = MultilabelAUROC(num_labels=args.nb_classes, average="macro", thresholds=None)
    # auc = ml_auroc(torch.cat(preds), torch.cat(trues))
    # metric_logger.meters['auc'].update(auc)  # Update the AUC meter

    print('* Acc@1 {top1.global_avg:.3f}'
          .format(top1=metric_logger.acc1))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}