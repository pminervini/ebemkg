#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

import os
import numpy as np
import cPickle as pickle
import rdflib

import sys
import getopt
import logging

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import pandas
import seaborn as sns

name_prefix = 'aifb'


def mystr(num):
    pow = int(np.log10(num))
    return str('10^{%s}' % pow)

def process(path, loss_threshold):
    logging.info('Processing %s ..' % (path))
    f = open(path, 'rb')
    document = pickle.load(f)

    if 'state' not in document.keys():
        return None

    name, line = '', None

    state = document['state']

    print('State: %s' % state)

    method = state['method']

    lr = state['lremb']
    decay = state['decay']
    ndim = state['ndim']
    aeps = state['epsilon']
    max_lr = state['max_lr']

    method_name = method.upper()

    if method_name == 'ADADELTA':
        name = method_name + ' $(1 - \\rho) = ' + mystr(1.0 - float(decay)) + ', \\epsilon=' + mystr(float(aeps)) + '$'
    elif method_name == 'RMSPROP':
        name = method_name + ' $(1 - \\rho) = ' + mystr(1.0 - float(decay)) + ', \\eta=' + mystr(float(lr)) + ', \\omega=' + mystr(float(max_lr)) + '$'
    else:
        name = method_name + ' $\\eta = ' + mystr(float(lr)) + '$'

    logging.info(name)

    if 'average_costs_per_epoch' in document.keys():
        average_costs_per_epoch = document['average_costs_per_epoch']

        # [ (mean, std), (mean, std), (mean, std), .. ]
        costs_per_epoch = [(np.mean(epoch_costs), np.std(epoch_costs))
                           for epoch_costs in average_costs_per_epoch]
        print(name + ' = ' + str(['{:.3f}'.format(mean_cost)
                                  for (mean_cost, _) in costs_per_epoch]).replace('\'', ''))

        mean_costs = [mean_cost for (mean_cost, _) in costs_per_epoch]

        # linestyle or ls	[ '-' | '--' | '-.' | ':' | 'steps' | ...]
        linestyle = '-'

        if method == 'ADAGRAD':
            linestyle = '--'
        elif method == 'ADADELTA':
            linestyle = ':'
            if aeps > 1e-4:
                linestyle = '-.'
        elif method == 'RMSPROP':
            linestyle = '-'


    # FILTER - let's check the last value of mean_costs: if it's > 100, we do not print it
    ret = None
    if (loss_threshold is None) or (mean_costs[-1] < loss_threshold):
        #epochs = pandas.Series(range(1, len(mean_costs) + 1), name='epoch')
        #costs = pandas.Series(mean_costs, name=name)
        #line = sns.tsplot(costs, time=epochs, value='cost');

        line = plt.plot(mean_costs, label=name, linestyle=linestyle)
        ret = (line, name)

    f.close()

    return ret


def main(argv):

    for arg in argv:
        if (arg == '-h' or arg == '--help'):
            logging.info('Sample usage: LOSS_THRESHOLD=200 SAVE_FILE=aifb_adadelta_rescaled_200.png %s ~/models/*.pkl' % (sys.argv[0]))
            return

    line_names = []

    # beautifying with seaborn
    sns.set_style('white')
    sns.despine(trim=True)
    sns.set_context('poster')

    loss_threshold = None
    if ('LOSS_THRESHOLD' in os.environ.keys()):
        loss_threshold = float(os.environ['LOSS_THRESHOLD'])

    for arg in argv:
        res = process(arg, loss_threshold)
        logging.info(res)
        if res is not None:
            line_names += [res]

    logging.info('line names (%d): %s' % (len(line_names), line_names))

    sns.set_palette('hls', n_colors=len(line_names))

    ax = plt.subplot(111)

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=4)

    if loss_threshold is not None:
        plt.ylim(0, loss_threshold + 10)

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    plt.grid()

    save_file = 'out.png'
    if ('SAVE_FILE' in os.environ.keys()):
        save_file = os.environ['SAVE_FILE']

    fig.savefig(save_file, additional_artists=[legend], dpi=100, bbox_inches='tight')

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
