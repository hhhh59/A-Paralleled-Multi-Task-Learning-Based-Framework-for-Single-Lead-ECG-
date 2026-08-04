import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch.nn.functional as F
import wfdb
import re
from os.path import exists


import torch
import torch.backends.cudnn as cudnn
# from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
from scipy.io import loadmat

import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

from engine_pretrain import train_one_epoch
from utils.datasets_h5 import CustomDataset, CustomDataset_Single_lead
from torchmetrics.classification import MultilabelAUROC
from timm.utils import accuracy
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import pandas as pd
from ecg_preprocessing import Rwave_Detection
import matplotlib
matplotlib.use('Agg')
num_categories = 3

noise_criterion = torch.nn.MSELoss()
threshold1 = 0.8
threshold2 = 0.4

def add_right_cax(ax, pad, width):
    '''
    在一个ax右边追加与之等高的cax.
    pad是cax与ax的间距.
    width是cax的宽度.
    '''
    axpos = ax.get_position()
    caxpos = matplotlib.transforms.Bbox.from_extents(
        axpos.x1 + pad,
        axpos.y0,
        axpos.x1 + pad + width,
        axpos.y1
    )
    cax = ax.figure.add_axes(caxpos)

    return cax

def scaled_coefficient(clean_ecg_energy, noise, target_snr):
    # clean_ecg_energy = signal_energy[0,0]
    # noise = em
    # target_snr = 10
    adjusted_noise_energy = clean_ecg_energy / (10 ** (target_snr / 10))
    noise_energy = np.sum(noise ** 2)
    coefficient = np.sqrt(adjusted_noise_energy / noise_energy)
    return coefficient

def calculate_snr(signal, noise):
    signal_energy = np.sum(signal ** 2)
    noise_energy = np.sum(noise ** 2)
    snr = 10 * np.log10(signal_energy / noise_energy)
    return snr

def add_various_noise(signal, status):
    # signal = np.random.normal(0, 1, (64, 1, 1, 1000))

    status = 'train'
    assert status in ['train', 'test', 'val']
    type_map = {'train': 2880, 'val': 360, 'test': 362}
    len = type_map[status]
    signal = signal.detach().numpy()
    signal_energy = np.sum(signal ** 2, axis=-1)
    label = np.random.randint(0, 3, size=signal.shape[0])
    if signal_energy<0.01:
        print("ECG is too weak")
        label = np.array([2])
    # gaussian_noise = np.random.normal(0, 1, signal.shape)
    # noise_energy = np.sum(gaussian_noise ** 2, axis=-1).squeeze()
    noisy_signal = np.zeros(signal.shape)
    random_snr = np.zeros(label.shape)
    for i, l in enumerate(label):
        if l == 0:
            random_snr[i] = np.random.randint(18, 40)
        elif l == 1:
            random_snr[i] = np.random.randint(5, 18)
        elif l == 2:
            random_snr[i] = np.random.randint(-10, 5)

        mixed = np.random.randint(1,4)
        if mixed == 1:
            name = 'em'  + str(np.random.randint(1, len+1)) + '.mat'
            path = '.\\data_h5\\noise_Data_' + str(status) + '\\' + name
            em = loadmat(path)['eletrode_artifacts']; em = np.transpose(em)
            coefficient = scaled_coefficient(signal_energy[i], em, random_snr[i])
            noisy_signal[i] = signal[i] + coefficient * em
        elif mixed == 2:
            name = 'ma'  + str(np.random.randint(1, len+1)) + '.mat'
            path = '.\\data_h5\\noise_Data_' + str(status) + '\\' + name
            ma = loadmat(path)['EMG']; ma = np.transpose(ma)
            coefficient = scaled_coefficient(signal_energy[i], ma, random_snr[i])
            noisy_signal[i] = signal[i] + coefficient * ma
        elif mixed == 3:
            name = 'em' + str(np.random.randint(1, len + 1)) + '.mat'
            path = '.\\data_h5\\noise_Data_' + str(status) + '\\' + name
            em = loadmat(path)['eletrode_artifacts'];
            em = np.transpose(em)
            name = 'ma' + str(np.random.randint(1, len + 1)) + '.mat'
            path = '.\\data_h5\\noise_Data_' + str(status) + '\\' + name
            ma = loadmat(path)['EMG'];
            ma = np.transpose(ma)
            coefficient = scaled_coefficient(signal_energy[i], em + ma, random_snr[i])
            noisy_signal[i] = signal[i] + coefficient * ma + coefficient * em

    noisy_signal_energy = np.sum(noisy_signal ** 2, axis=-1)
    if noisy_signal_energy<0.05:
        print("ECG is too weak")
    return torch.tensor(noisy_signal), label, random_snr


def unpatchify(x):
    """
    x: (N, L, patch_size_height*patch_size_width*1)
    imgs: (N, 1, H, W) - 12 channel ECG - H = No. of channels, W = Length of ECG signal (1000 in this case)
    """
    ph = 1
    pw = 20

    # h = w = int(x.shape[1]**.5)
    # assert h * w == x.shape[1]
    h = 1
    w = x.shape[1]//1

    x = x.reshape(shape=(h, w, ph, pw, 1))
    x = torch.einsum('hwpqc->chpwq', x)
    imgs = x.reshape(shape=(1, h * ph, w * pw))
    return imgs


def test_multi_task(data_loader, model, device, clean=False):
    criterion = torch.nn.BCEWithLogitsLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    auc_meter = misc.SmoothedValue(window_size=1, fmt='{avg:.4f}')

    metric_logger.add_meter('auc', auc_meter)  # Add this line
    auc_meter.update(0)
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    # trues = []
    # preds = []
    # true_class_list = []
    # pred_class_list = []
    # score_list = []
    # ind_list = []
    # pic_num = 0
    k = 0
    for (samples, ind) in metric_logger.log_every(data_loader, 1, header):
        # if k == 370:
        #     print("Adding Gaussian Noise")
        k += 1
        target_ecg = samples

        if clean:
            label = np.array([0] * samples.shape[0])
        else:
            samples, label, snr = add_various_noise(samples, 'test')


        # samples = samples.half()
        targets = torch.zeros(label.shape[0], num_categories, dtype=torch.float32)
        targets[np.arange(label.shape[0]), label.astype(int)] = 1

        # if samples.shape[0] == 1:
        #     targets = torch.zeros(num_categories)
        #     if score < 3:
        #         targets[0] = 1
        #     elif score >=3 and score <= 5:
        #         targets[1] = 1
        #     elif score >=6 and score <= 9:
        #         targets[2] = 1
        #     elif score >=10 and score <= 12:
        #         targets[3] = 1
        #     elif score >=13:
        #         targets[4] = 1
        #     targets = torch.unsqueeze(targets, 0)
        # else:
        #     score[np.where(score<3)[0]] = 0
        #     score[np.where((score>=3) & (score<=5))[0]] = 1
        #     score[np.where((score>=6) & (score<=9))[0]] = 2
        #     score[np.where((score >= 10) & (score <=12))[0]] = 3
        #     score[np.where(score >= 13)[0]] = 4
        #     targets = torch.zeros(score.shape[0], num_categories, dtype=torch.float32)
        #     targets[np.arange(score.shape[0]), score.astype(int)] = 1

        # compute output
        with torch.amp.autocast('cuda'):
            denoise_loss, pred, output = model(samples, target_ecg)
            loss = criterion(output, targets)

        acc1 = accuracy_score(targets, torch.sigmoid(output) > 0.5)*100
        # ml_auroc = MultilabelAUROC(num_labels=args.nb_classes, average="macro", thresholds=None)
        # auc = ml_auroc(torch.sigmoid(output.cpu()), target.cpu().int())
        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1, n=batch_size)
        # metric_logger.meters['auc'].update(auc, n=batch_size)
        # trues.append(targets.int())
        # preds.append(torch.sigmoid(output))
        true_class = np.where(targets.detach().cpu().numpy() == 1)[1]
        # pred_class = np.where(torch.sigmoid(output) > 0.5)[1]
        pred_class = torch.argmax(torch.sigmoid(output), dim=1).detach().cpu().numpy()
        # true_class_list.append(true_class)
        # pred_class_list.append(pred_class)
        # score_list.append(score) # median of snr
        # ind_list.append(ind.detach().cpu().numpy()[0])
        # if acc1 <0.8:
            # np.save(samples.detach().cpu().numpy(), './output_dir_5classes_1st/' + str(ind.detach().cpu().numpy()))

        ecg_data = unpatchify(pred).detach().numpy()

        '''
        # 12-lead figure
        if True:
            # pic_num += 1
            # figure
            for j in range(batch_size):
                time_axis = np.arange(0, 1000)
                fig, axes = plt.subplots(12, 2, figsize=(24, 18), sharex=True,
                                         gridspec_kw={'wspace': 0.2, 'hspace': 0.5})
                for i in range(12):
                    axes[i,0].set_ylabel(str(random_snr[i]))
                for i in range(12):
                    input_channel = samples[0, 0, i, :]
                    ecg_channel = ecg_data[0, i, :]
                    target_channel = target_ecg[0, 0, i, :]
                    axes[i,0].plot(time_axis, input_channel, color='black', alpha=0.7)
                    axes[i,1].plot(time_axis, ecg_channel, color='blue', alpha=0.7)
                    axes[i,1].plot(time_axis, target_channel, color='red', alpha=0.7)

                axes[0,0].set_title('SNR=' + str(score) + '\n' +
                                  "Target Label: " + str(true_class[0]) + "  Predicted Label: "+ str(pred_class[0]))

                axes[0, 1].set_title('Reconstructed ECG Signals')
                fig.legend()
                plt.tight_layout()
                for i in range(12):
                    axes[i,0].legend().set_visible(False)
                    axes[i, 1].legend().set_visible(False)
                plt.savefig('./output_dir/' + str(ind.detach().cpu().numpy()[0]) +
                            '_' + str(true_class[0]) + '_' + str(pred_class[0]) + '_' + str(score) + '.png')
                # plt.show()
                plt.close()
        '''
        # snr=np.array([[0]])
        # pic_num += 1
        # figure
        for j in range(batch_size):
            time_axis = np.arange(0, 1000)
            fig, axes = plt.subplots(2, 1, figsize=(25, 15), sharex=True,
                                     gridspec_kw={'wspace': 0.2, 'hspace': 0.5})



            input_channel = samples[0, 0, :, :]
            ecg_channel = ecg_data[0, :, :]
            target_channel = target_ecg[0, 0, :, :]
            axes[0].plot(time_axis, input_channel.squeeze(), color='black', alpha=0.7)
            axes[1].plot(time_axis, ecg_channel.squeeze(), color='blue', alpha=0.7)
            axes[1].plot(time_axis, target_channel.squeeze(), color='red', alpha=0.7)

            axes[0].set_title('SNR=' + str(int(snr.squeeze())) + '\n' +
                                 "Target Label: " + str(true_class[0]) + "  Predicted Label: " + str(pred_class[0]))

            axes[1].set_title('Reconstructed ECG Signals')
            fig.legend()
            plt.tight_layout()

            plt.savefig('./output_dir/' + str(ind.detach().cpu().numpy()[0]) +
                        '_' + str(true_class[0]) + '_' + str(pred_class[0]) + '_' + str(int(snr.squeeze())) + '.png')
            # plt.show()
            plt.close()


    # df = pd.DataFrame({
    #     'SNR': score_list,
    #     'Target Label': true_class_,
    #     'Predicted Label': pred_class_
    # })
    # df.to_csv('./output_dir_5classes_1st/test_results.csv', index=False)
    # metric_logger.synchronize_between_processes()
    ml_auroc = MultilabelAUROC(num_labels=num_categories, average="macro", thresholds=None)
    # auc = ml_auroc(torch.cat(preds), torch.cat(trues))
    # metric_logger.meters['auc'].update(auc)  # Update the AUC meter

    # print('* Acc@1 {top1.global_avg:.3f} auc {aucs:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, aucs = auc, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def test_multi_task_BUT(data_loader, model, device, clean=False):
    criterion = torch.nn.BCEWithLogitsLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    auc_meter = misc.SmoothedValue(window_size=1, fmt='{avg:.4f}')

    metric_logger.add_meter('auc', auc_meter)  # Add this line
    auc_meter.update(0)
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    # trues = []
    # preds = []
    # true_class_list = []
    # pred_class_list = []
    # score_list = []
    # ind_list = []
    # pic_num = 0
    k = 0
    for (samples, ind) in metric_logger.log_every(data_loader, 1, header):
        # if k == 370:
        #     print("Adding Gaussian Noise")
        # k += 1
        target_ecg = samples[:,:,:,:-1]
        label = np.array(samples[:,:,:,-1])
        # samples = samples.half()
        targets = torch.zeros(label.shape[0], num_categories, dtype=torch.float32)
        targets[np.arange(label.shape[0]), label.astype(int)] = 1
        label = label.squeeze(-1).squeeze(-1)
        label = torch.tensor(label)
        # compute output
        with torch.amp.autocast('cuda'):
            # output,l, denoise_loss, pred = model(target_ecg, target_ecg, label)
            output,l, denoise_loss, pred = model(target_ecg)
            l = torch.sigmoid(l.squeeze()).to(torch.float32)
            # output = torch.zeros(targets.shape)
            level = torch.mean(l).item()
            # if level >= threshold1:
            #     output[0,0] = 1
            # elif level < threshold2:
            #     output[0,2] = 1
            # else:
            #     output[0,1] = 1
            loss = criterion(output, targets)
            # noiseLabel_loss = noise_criterion(label, noiseLabel)
        acc1 = accuracy_score(targets, torch.sigmoid(output) > 0.5)*100
        # ml_auroc = MultilabelAUROC(num_labels=args.nb_classes, average="macro", thresholds=None)
        # auc = ml_auroc(torch.sigmoid(output.cpu()), target.cpu().int())
        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1, n=batch_size)
        # metric_logger.meters['auc'].update(auc, n=batch_size)
        # trues.append(targets.int())
        # preds.append(torch.sigmoid(output))
        true_class = np.where(targets.detach().cpu().numpy() == 1)[1]
        # pred_class = np.where(torch.sigmoid(output) > 0.5)[1]
        pred_class = torch.argmax(torch.sigmoid(output), dim=1).detach().cpu().numpy()
        confidence = torch.sigmoid(output).detach().cpu().numpy()
        confidence = np.round(confidence, decimals=2)
        # true_class_list.append(true_class)
        # pred_class_list.append(pred_class)
        # score_list.append(score) # median of snr
        # ind_list.append(ind.detach().cpu().numpy()[0])
        # if acc1 <0.8:
            # np.save(samples.detach().cpu().numpy(), './output_dir_5classes_1st/' + str(ind.detach().cpu().numpy()))
        # ecg_data = unpatchify(pred).detach().numpy()
        ecg_data = pred.detach().numpy()
        snr = calculate_snr(ecg_data, target_ecg.detach().numpy() - ecg_data)
        snr = snr.round(0)
        # figure
        # if true_class[0] != pred_class[0]:
        if True:
        #  =========================================================================展示噪声分布=========================
        # if ind.detach().cpu().numpy()[0] ==260:
            for j in range(batch_size):
                time_axis = np.arange(0, 1000)
                fig, axes = plt.subplots(3, 1, figsize=(30, 15), sharex=True,
                                         gridspec_kw={'wspace': 0.2, 'hspace': 0.18})
                plt.subplots_adjust(left=0.03, right=0.92, bottom=0.1, top=0.96)
                input_channel = samples[0, 0, :, :-1]
                ecg_channel = ecg_data[0, :, :]
                # target_channel = target_ecg[0, 0, :, :]
                noiseLabel = l
                # r_pos_before = Rwave_Detection(input_channel.squeeze(), 200)
                try :
                    r_pos_before = Rwave_Detection(input_channel.squeeze(), 200)
                    axes[0].scatter(r_pos_before, input_channel[0][r_pos_before]/2, marker='o', color='cyan', s=100)
                    r_pos_after = Rwave_Detection(ecg_channel.squeeze(), 200)
                    axes[1].scatter(r_pos_after, ecg_channel[0][0][r_pos_after]/2, marker='o', color='cyan', s=100)
                except IndexError as e:
                    pass
                except ValueError as e:
                    pass
                except Exception as r:
                    pass
                finally:
                    pass
                axes[0].plot(time_axis, (input_channel.squeeze())/2, color='black', alpha=0.7, linewidth=3)
                axes[0].set_xlim(0, 1000)
                axes[0].set_xticks(np.arange(0, 1000, 40))
                axes[0].set_yticks(np.arange(np.floor(np.min(ecg_channel/2)), np.ceil(np.max(ecg_channel/2))+1, 1.0))
                axes[0].minorticks_on()
                axes[0].xaxis.set_minor_locator(AutoMinorLocator(5))
                axes[0].grid(which='major', linestyle='-', linewidth='0.7', color='red')
                axes[0].grid(which='minor', linestyle='-', linewidth='0.5', color=(1, 0.7, 0.7))
                axes[0].tick_params(axis='x', labelbottom=False)  # 隐藏 x 轴标签
                axes[0].tick_params(axis='y', labelleft=False)

                axes[1].plot(time_axis, (ecg_channel.squeeze())/2, color='blue', alpha=0.7, linewidth=3)
                axes[1].set_xticks(np.arange(0, 1000, 40))
                axes[1].set_yticks(np.arange(np.floor(np.min(ecg_channel/2)), np.ceil(np.max(ecg_channel/2))+1, 1.0))
                axes[1].minorticks_on()
                axes[1].xaxis.set_minor_locator(AutoMinorLocator(5))
                axes[1].grid(which='major', linestyle='-', linewidth='0.7', color='red')
                axes[1].grid(which='minor', linestyle='-', linewidth='0.5', color=(1, 0.7, 0.7))
                axes[1].tick_params(axis='x', labelbottom=False)  # 隐藏 x 轴标签
                axes[1].tick_params(axis='y', labelleft=False)

                # axes[1].scatter(r_pos_after, ecg_channel[0][0][r_pos_after], marker='o', color='cyan', s=80)
                # axes[1].plot(time_axis, target_channel.squeeze(), color='red', alpha=0.7)
                # axes[1].plot(time_axis, noiseLabel.detach().numpy()*10, color='green', alpha=0.7)

                # axes[0].set_title( "WD " + str(ind.detach().cpu().numpy()[0]+1) + " Target Label: " + str(true_class[0]) + "  Predicted Label: " + str(pred_class[0]),
                #                   fontdict={'family': 'Times New Roman', 'weight':'bold', 'size': 30})
                # axes[1].set_title('Reconstructed ECG Signals', fontdict={'family': 'Times New Roman', 'size': 30})
                axes[0].set_title( "Original ECG",fontdict={'family': 'Times New Roman', 'size': 40})
                axes[1].set_title('Denoised ECG', fontdict={'family': 'Times New Roman', 'size': 40})
                # x = np.arange(1000)
                # probabilities = noiseLabel.detach().numpy()
                # raw = input_channel.detach().numpy()
                # sc = axes[2].scatter(x, raw, c=probabilities, cmap='viridis', s=260, edgecolors='black', vmin=0, vmax=1)
                # # plt.axes[2].colorbar(sc, label='Probability')
                # cax = add_right_cax(axes[2], pad=0.02, width=0.02)
                # cb = fig.colorbar(sc, cax=cax)
                # cb.set_label('Probability', fontdict={'family': 'Times New Roman', 'size': 40})
                # # fig.colorbar(sc, ax=axes[2], orientation='horizontal', label='Probability')
                # axes[2].grid(True, linestyle='--', alpha=0.5)
                # # axes[2].set_ylim(0, 1)
                # axes[2].set_title('Noise Level Graph', fontdict={'family': 'Times New Roman', 'size': 40})
                # axes[2].tick_params(axis='x', labelbottom=False)  # 隐藏 x 轴标签
                # axes[2].tick_params(axis='y', labelleft=False)
                # # fig.suptitle('(a) No.227 of WD',
                # #              y=0.05,  # 位置在底部2%处
                # #              fontsize=40,
                # #              fontfamily='Times New Roman')
                # if ind.detach().cpu().numpy()[0] ==227:
                #     fig.suptitle('(a) No.227 of WD',
                #                  y=0.05,  # 位置在底部2%处
                #                  fontsize=40,
                #                  fontfamily='Times New Roman')
                # if ind.detach().cpu().numpy()[0] ==425:
                #     fig.suptitle('(b) No.425 of WD',
                #                  y=0.05,  # 位置在底部2%处
                #                  fontsize=40,
                #                  fontfamily='Times New Roman')
                # if ind.detach().cpu().numpy()[0] ==260:
                #     fig.suptitle('(c) No.260 of WD',
                #                  y=0.05,  # 位置在底部2%处
                #                  fontsize=40,
                #                  fontfamily='Times New Roman')

                plt.savefig('./output_dir_v5_9519/' + str(ind.detach().cpu().numpy()[0]+0) +
                # plt.savefig('./output_dir_v5_1/' + str(ind.detach().cpu().numpy()[0]+8001) +
                # plt.savefig('./output_dir/' + str(ind.detach().cpu().numpy()[0]+8001) +
                            '_' + str(true_class[0]) + '_' + str(pred_class[0]) +
                            '_' + str(int(snr.squeeze())) +
                            '_' + str(round(level,4)) +
                            '_' + str(confidence[0][0]) +
                            '_' + str(confidence[0][1]) +
                            '_' + str(confidence[0][2]) +
                            '.png')
                plt.close()
        # #  展示Rwave=====================================================================================================
        # if ind.detach().cpu().numpy()[0] == 220:
        #     for j in range(batch_size):
        #         time_axis = np.arange(0, 1000)
        #         fig, axes = plt.subplots(1, 1, figsize=(30, 5), sharex=True,
        #                                  gridspec_kw={'wspace': 0.2, 'hspace': 0.5})
        #         plt.subplots_adjust(left=0.03, right=0.97, bottom=0.25, top=0.96)
        #         input_channel = samples[0, 0, :, :-1]
        #         ecg_channel = ecg_data[0, :, :]
        #
        #         axes.set_xlim(0, 1000)
        #         axes.plot(time_axis, (input_channel.squeeze())/2, color='black', alpha=0.4, label='Original ECG', linewidth=3)
        #         axes.plot(time_axis, (ecg_channel.squeeze())/2, color='blue', alpha=0.7, label='Denoised ECG', linewidth=3)
        #         axes.set_xticks(np.arange(0, 1000, 40))
        #         axes.set_yticks(np.arange(np.floor(np.min(ecg_channel/2)), np.ceil(np.max(ecg_channel/2))+1, 1.0))
        #         axes.minorticks_on()
        #         axes.xaxis.set_minor_locator(AutoMinorLocator(5))
        #         axes.grid(which='major', linestyle='-', linewidth='0.7', color='red')
        #         axes.grid(which='minor', linestyle='-', linewidth='0.5', color=(1, 0.7, 0.7))
        #
        #
        #         r_pos_before = Rwave_Detection(input_channel.squeeze(), 200)
        #         axes.scatter(r_pos_before, input_channel[0][r_pos_before]/2, marker='o', color='cyan', s=300,label='R-peaks before')
        #         r_pos_after = Rwave_Detection(ecg_channel.squeeze(), 200)
        #         axes.scatter(r_pos_after, ecg_channel[0][0][r_pos_after]/2, marker='o', color='magenta', s=300,label='R-peaks after')
        #
        #         axes.tick_params(axis='x', labelbottom=False)  # 隐藏 x 轴标签
        #         axes.tick_params(axis='y', labelleft=False)
        #         # axes.set_title( "WD " + str(ind.detach().cpu().numpy()[0]+2001),
        #         #                   fontdict={'family': 'serif', 'weight': 'bold', 'size': 15})
        #         # axes[1].set_title('Reconstructed ECG Signals', fontdict={'family': 'serif', 'weight': 'bold', 'size': 30})
        #         # axes.legend(loc='upper right',framealpha=1)
        #
        #         # fig.suptitle('(a) No.227 of WD', y=0.15,  # 位置在底部2%处
        #         #              fontsize=40, fontfamily='Times New Roman')
        #         if ind.detach().cpu().numpy()[0] ==220:
        #             fig.suptitle('(a) No.220 of WD', y=0.15,  # 位置在底部2%处
        #                          fontsize=50, fontfamily='Times New Roman')
        #         if ind.detach().cpu().numpy()[0] ==227:
        #             fig.suptitle('(b) No.227 of WD', y=0.15,  # 位置在底部2%处
        #                          fontsize=50, fontfamily='Times New Roman')
        #         if ind.detach().cpu().numpy()[0] ==255:
        #             fig.suptitle('(c) No.255 of WD', y=0.15,  # 位置在底部2%处
        #                          fontsize=50, fontfamily='Times New Roman')
        #         if ind.detach().cpu().numpy()[0] ==265:
        #             fig.suptitle('(d) No.265 of WD', y=0.15,  # 位置在底部2%处
        #                          fontsize=50, fontfamily='Times New Roman')
        #         if ind.detach().cpu().numpy()[0] ==300:
        #             fig.suptitle('(e) No.300 of WD', y=0.15,  # 位置在底部2%处
        #                          fontsize=50, fontfamily='Times New Roman')
        #         plt.savefig('./output_dir_v5_9519/' + str(ind.detach().cpu().numpy()[0]+0) +
        #                     '_' + str(true_class[0]) + '_' + str(pred_class[0]) +
        #                     '.png')
        #         plt.close()
    ml_auroc = MultilabelAUROC(num_labels=num_categories, average="macro", thresholds=None)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def test_multi_task_model():
    from models_multi_task import mae_vit_1dcnn
    test_model = mae_vit_1dcnn()
    data_path = './data_h5/ptb_200Hz_lead1.h5'
    # checkpoint_loc = './output_dir_multi_task_snr_normal_test_unclean/checkpoint-80.pth'
    checkpoint_loc = './output_dir/checkpoint-100.pth'
    clean = False

    data_split = 0.8
    seed = 0 + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    full_dataset = CustomDataset(data_path)  # Training Data -
    train_size = int(data_split * len(full_dataset))
    test_size = len(full_dataset) - train_size
    dataset_train, dataset_test = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    del full_dataset, dataset_train
    # val_size = int(len(full_dataset) * 0.1)
    # dataset_train, dataset_val
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        drop_last=False
    )
    # Reconstruction from the 220th epoch checkpoints
    test_model = test_model.double()
    test_model.eval()
    checkpoint = torch.load(checkpoint_loc, map_location='cuda')
    test_model.load_state_dict(checkpoint['model'], strict=True)

    test_stats = test_multi_task(data_loader_test, test_model, torch.device('cuda'), clean=clean)
    print(f"Accuracy of the network on the {len(dataset_test)} test ECG: {test_stats['acc1']:.1f}%")

def test_multi_task_model_BUT():
    from models_noiselabel_v5_9519 import mae_vit_1dcnn
    from models_noiselabel_v3 import mae_vit_1dcnn
    # from models_MIL_92 import mae_vit_1dcnn
    test_model = mae_vit_1dcnn()
    # data_path = './data_h5/LAB0.h5'
    # data_path = './data_h5/LAB3001.h5'
    # data_path = './data_h5/LAB_clean.h5'
    # data_path = './data_h5/v3_test.h5'
    # data_path = './data_h5/PTB_train.h5'
    # data_path = './data_h5/arrh_train.h5'
    # data_path = './data_h5/BUT_v1.h5'
    # data_path = './data_h5/ptbxl_all.h5'
    # data_path = './data_h5/BUT_2.h5'

    # data_path = './data_h5/BUT_clean_mv.h5'
    data_path = './data_h5/WD.h5'
    data_path = './data_h5/BUT_v1.h5'
    # data_path = './data_h5/PTB_aug.h5'
    # checkpoint_loc = './output_dir_multi_task_snr_normal_test_unclean/checkpoint-80.pth'
    checkpoint_loc = './output_dir_v5_9519/checkpoint-49.pth'
    checkpoint_loc = './output_dir/checkpoint-99.pth'
    # checkpoint_loc = './output_dir_v3_9454/checkpoint-38.pth'
    # checkpoint_loc = './output_dir_v5_9545/checkpoint-40.pth'
    # checkpoint_loc = './output_dir_aug/checkpoint-40.pth'
    # checkpoint_loc = './output_dir_v5_1/checkpoint-39.pth'
    # checkpoint_loc = './output_dir_overlap/checkpoint-79.pth'
    clean = False

    # data_split = 0.8
    seed = 0 + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    full_dataset = CustomDataset_Single_lead(data_path)  # Training Data -
    # train_size = int(data_split * len(full_dataset))
    # test_size = len(full_dataset) - train_size
    # dataset_train, dataset_test = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    # del full_dataset, dataset_train
    # val_size = int(len(full_dataset) * 0.1)
    # dataset_train, dataset_val
    sampler_test = torch.utils.data.SequentialSampler(full_dataset)
    data_loader_test = torch.utils.data.DataLoader(
        full_dataset, sampler=sampler_test,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        drop_last=False
    )
    # Reconstruction from the 220th epoch checkpoints
    test_model = test_model.double()
    test_model.eval()
    checkpoint = torch.load(checkpoint_loc, map_location='cuda')
    test_model.load_state_dict(checkpoint['model'], strict=True)

    test_stats = test_multi_task_BUT(data_loader_test, test_model, torch.device('cuda'), clean=clean)
    print(f"Accuracy of the network on the {len(full_dataset)} test ECG: {test_stats['acc1']:.1f}%")
# =========================================================================================================
if __name__ == '__main__':

    # test_classification_model()
    # test_multi_task_model()
    test_multi_task_model_BUT()
