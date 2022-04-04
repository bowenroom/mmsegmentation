# import the packages to read the json file 
#%%
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
from fastai.basics import *
from fastai.vision import *
import argparse
import json
from collections import defaultdict

#%%

def plot_curve(log_dicts, args):
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)
    # if legend is None, use {filename}_{key} as legend
    legend = args.legend
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            for metric in args.keys:
                legend.append(f'{json_log}_{metric}')
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys

    num_metrics = len(metrics)
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        for j, metric in enumerate(metrics):
            print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
            plot_epochs = []
            plot_iters = []
            plot_values = []
            # In some log files, iters number is not correct, `pre_iter` is
            # used to prevent generate wrong lines.
            pre_iter = -1
            for epoch in epochs:
                epoch_logs = log_dict[epoch]
                if metric not in epoch_logs.keys():
                    continue
                if metric in ['mIoU', 'mAcc', 'aAcc']:
                    plot_epochs.append(epoch)
                    plot_values.append(epoch_logs[metric][0])
                else:
                    for idx in range(len(epoch_logs[metric])):
                        if pre_iter > epoch_logs['iter'][idx]:
                            continue
                        pre_iter = epoch_logs['iter'][idx]
                        plot_iters.append(epoch_logs['iter'][idx])
                        plot_values.append(epoch_logs[metric][idx])
            ax = plt.gca()
            label = legend[i * num_metrics + j]
            if metric in ['mIoU', 'mAcc', 'aAcc']:
                ax.set_xticks(plot_epochs)
                plt.xlabel('epoch')
                plt.plot(plot_epochs, plot_values, label=label, marker='o')
            else:
                plt.xlabel('iter')
                plt.plot(plot_iters, plot_values, label=label, linewidth=0.5)
        plt.legend()
        if args.title is not None:
            plt.title(args.title)
    if args.out is None:
        plt.show()
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out)
        plt.cla()


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    parser.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['mIoU'],
        help='the metric that you want to plot')
    parser.add_argument('--title', type=str, help='title of figure')
    parser.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    return args


def load_json_logs(json_logs):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts = [dict()]
    log_dict = dict()
    with open(json_logs, 'r') as log_file:
        for line in log_file:
            log = json.loads(line.strip())
            # skip lines without `epoch` field
            if 'epoch' not in log:
                continue
            epoch = log.pop('epoch')
            if epoch not in log_dict:
                log_dict[epoch] = defaultdict(list)
            for k, v in log.items():
                log_dict[epoch][k].append(v)
    return log_dict

#%%
# load log file 
# log_dicts = load_json_logs('/home/ubuntu/paperCode/codeLib/mmsegmentation/work_dirs/fcn_hr18_4x4_512x512_80k_vaihingen/20220303_091426.log.json')
def getJsonDict(path):
    # read a json log file:
    with open(path, 'r') as log_file:
        log_dicts = [json.loads(line.strip()) for line in log_file]
        temp = []
        for log in log_dicts:
            # extract all the val epochs
            if next(iter(log)) != 'env_info' and  log['mode'] == 'val':
                temp.append(log)
    print(f'number of the val epochs is:{len(temp)}')
    return temp
    # print(temp)
# %%
# exTractMetrics
def exTractMetrics(logDict):
    singleF = []
    FscoreF = []
    PrecisionF = []
    RecallF = []
    IoUF = []
    AccF = []
    # iterate the list
    for i,log in enumerate(logDict):
        # extract the 'epoch'
        epoch = log['epoch']
        log.pop('epoch')
        single = {}
        Fscore = {}
        Precision = {}
        Recall = {}
        IoU = {}
        Acc = {}
        for k,v in log.items():
            single['epoch'] = epoch
            Fscore['epoch'] = epoch
            Precision['epoch'] = epoch
            Recall['epoch'] = epoch
            IoU['epoch'] = epoch
            Acc['epoch'] = epoch
            
            if '.' not in k:
                single[k] = v
            elif 'Fscore' in k:
                Fscore[k] = v
            elif 'Precision' in k:
                Precision[k] = v
            elif 'Recall' in k:
                Recall[k] = v
            elif 'IoU' in k:
                IoU[k] = v
            elif 'Acc' in k:
                Acc[k] = v
            
        singleF.append(single)
        FscoreF.append(Fscore)
        PrecisionF.append(Precision)
        RecallF.append(Recall)
        IoUF.append(IoU)
        AccF.append(Acc)
    return (singleF,FscoreF,PrecisionF,RecallF,IoUF,AccF)
aa= exTractMetrics(getJsonDict('/home/ubuntu/paperCode/codeLib/mmsegmentation/work_dirs/fcn_hr18_4x4_512x512_80k_vaihingen/20220303_091426.log.json'))
# %%
# draw the curve
def drawCurve(metrics):

    # metrics is tuple of list
    # metrics contain 6 metric, each metric is a list of dict
    # plot the curve of each metric
    # x axis is epoch
    # y axis is metric value
    plt.figure()
    for i, metric in enumerate(metrics):
        for j, metric_dict in enumerate(metric):
            # pop the 'mode'=val
            if 'mode' in metric_dict:
                metric_dict.pop('mode')
                metric_dict.pop('iter')
                metric_dict.pop('lr')
            epoch_value = list(metric_dict.values())[0]

            print(epoch_value)
            print('-----------')
drawCurve(aa)




