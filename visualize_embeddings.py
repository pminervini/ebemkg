#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

import os
import sys
import getopt
import logging
import random
import heapq
import numpy

import matplotlib
import matplotlib.pyplot as plt

import pandas
import seaborn as sns

import data.util as util
import persistence.layer as persistence
import dimensionality.manifold as manifold
import dimensionality.cluster as cluster
import bson.objectid as objectid

import cPickle as pickle

uri_names = True

def show_heat(experiment, classes_dict, save=None):
    show_experiment(experiment)

    best = experiment['best_on_validation']

    entities, predicates = best['entities'], best['predicates']
    variable = 'Eemb'

    NE = len(entities)

    classes, elements = [], []
    for (_class, _elements) in classes_dict.items():
        for _element in _elements:
            classes += [_class]
            elements += [_element]

    indexes = [entities.index(element) for element in elements]
    class_idx = {
        _class: _idx for (_idx, _class) in enumerate(classes_dict.keys())}

    classes_numeric = [class_idx[_class] for _class in classes]

    logging.info('#Indexes: %d' % (len(indexes)))

    NI = len(indexes)

    label_palette = sns.color_palette("hls")

    sns.despine(trim=True)
    sns.set_context('poster')
    c_map = sns.blend_palette(["firebrick", "palegreen"], as_cmap=True)

    sns.set_style('white')

    parameter = best['parameters'][variable]

    embeds = numpy.asarray(parameter['value'])
    print("Embedding shape", embeds.shape)

    X = embeds.T

    Xr = X[numpy.asarray(indexes), :]

    sim_mat = Xr.dot(Xr.T)
    frame = pandas.DataFrame(sim_mat)

    heat = sns.heatmap(frame, linewidths=0, square=True, robust=True, xticklabels=False, yticklabels=False)

    for i, cl in enumerate(classes):
        if i > 0 and cl != classes[i - 1]:
            plt.axhline(len(classes) - i, c="w")
            plt.axvline(i, c="w")

    if save is None:
        plt.show(heat)
    else:
        plt.savefig(save)



def show_points(experiment, classes_dict, manifold_method, cluster_method, save=None):
    show_experiment(experiment)

    best = experiment['best_on_validation']

    entities, predicates = best['entities'], best['predicates']
    variable = 'Eemb'

    NE = len(entities)

    classes, elements = [], []
    for (_class, _elements) in classes_dict.items():
        for _element in _elements:
            classes += [_class]
            elements += [_element]

    indexes = [entities.index(element) for element in elements]
    class_idx = {
        _class: _idx for (_idx, _class) in enumerate(classes_dict.keys())}

    classes_numeric = [class_idx[_class] for _class in classes]

    logging.info('#Indexes: %d' % (len(indexes)))

    NI = len(indexes)

    sns.set_palette("hls")
    sns.set_context('poster')
    sns.set_style('white')

    marker_size = 10 #20 #200
    if (NI < 1000):
        marker_size = 200

    parameter = best['parameters'][variable]

    Xt = numpy.asarray(parameter['value'])
    X = numpy.transpose(Xt)

    Xr, Zr = X[numpy.asarray(indexes), :], None

    if cluster_method is not None:
        Zr = cluster_method.apply(Xr).tolist()
    else:
        Zr = numpy.asarray(classes_numeric)

    logging.info('Shape for Xr: %s', Xr.shape)

    Xr_2d = manifold_method.apply(Xr)

    Zr = classes
    aliases = {
        '<http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id1instance>': 'BusInfComm',
        '<http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id5instance>': 'UsabEng',
        '<http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id4instance>': 'CompManag',
        '<http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id2instance>': 'EffAlg',
        '<http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id3instance>': 'KnowManag',

        '<http://data.bgs.ac.uk/id/Lexicon/LithogeneticType/FLUV>': 'FLUV',
        '<http://data.bgs.ac.uk/id/Lexicon/LithogeneticType/GLACI>': 'GLACI'
    }
    #Zr = [aliases[_class] for _class in classes]

    if uri_names:
        Zr = [_class.split('/')[4].replace('>', '') for _class in classes]

    frame = pandas.DataFrame(dict(x=Xr_2d[:, 0], y=Xr_2d[:, 1], Genre=Zr))

    logging.info('Shape for X_2d: %s', Xr_2d.shape)
    logging.info('Number of entities: %d', NE)

    fg = sns.lmplot('x', 'y', data=frame, size=10, scatter_kws={'s': marker_size}, hue='Genre', fit_reg=False)

    if save is None:
        plt.show(fg)
    else:
        plt.savefig(save)


def show_experiment(experiment):
    for (key, value) in experiment.items():
        value_str = str(value)
        info = value_str[:75] + (value_str[75:] and '..')
        logging.info('Key: %s, Value: %s' % (key, info))


def main(argv):
    experiment_id = None
    classes_pkl = 'data/aifb/aifb_d2s_group_affiliates.pkl'
    is_heat = False

    k = None

    save_file = None

    manifold_name = 'TSNE'  # 'TSNE'
    cluster_name = None  # 'KMeans'

    conf = util.configuration()

    usage_str = ('Usage: %s [-h] [--experiment=<id>] [--classes=<classes.pkl>] [--manifold=<manifold>] [--cluster=<cluster>] [--top=<k>] [--heat] [--save=<out.png>]' % (sys.argv[0]))
    # Parse arguments
    try:
        opts, args = getopt.getopt(argv, 'h', ['experiment=', 'classes=', 'manifold=', 'cluster=', 'top=', 'heat', 'save='])
    except getopt.GetoptError:
        logging.warn(usage_str)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            logging.info(usage_str)

            logging.info('\t--experiment=<id> (default: %s)' % (experiment_id))
            logging.info('\t--classes=<classes.pkl> (default: %s)' % (classes_pkl))

            logging.info('\t--manifold=<manifold> (default: %s)' % (manifold_name))
            logging.info('\t--cluster=<cluster> (default: %s)' %  (cluster_name))
            logging.info('\t--top=<k> (default: %s)' %  (k))

            logging.info('\t--heat (show a heat map)')

            logging.info('\t--save=<out.png> (default: %s)' % (save_file))
            return

        if opt == '--experiment':
            experiment_id = arg
        if opt == '--classes':
            classes_pkl = arg

        if opt == '--manifold':
            manifold_name = arg
        if opt == '--cluster':
            cluster_name = arg
        if opt == '--top':
            k = int(arg)
        if opt == '--heat':
            is_heat = True

        if opt == '--save':
            save_file = arg

    # Allocating the persistence layer ..
    layer = persistence.PickleLayer(dir=conf.get('Persistence', 'path'))

    manifold_method, cluster_method = None, None

    if experiment_id is not None:
        manifold_class = getattr(manifold, manifold_name)
        manifold_method = manifold_class()

        if (cluster_name is not None):
            cluster_class = getattr(cluster, cluster_name)
            cluster_method = cluster_class()

        experiment = layer.get(experiment_id)
        classes_dict = pickle.load(open(classes_pkl, 'rb'))

        keys = list(classes_dict.keys())

        logging.info('Removing elements in the intersection ..')

        for i, _class in enumerate(keys):
            _albums = classes_dict[_class]
            _other_genres = keys[i+1:]
            for _other_genre in _other_genres:
                classes_dict[_class] = classes_dict[_class] - classes_dict[_other_genre]

        logging.info('.. done.')

        _classes = list(classes_dict.keys())
        _cardinalities = [len(classes_dict[_class]) for _class in _classes]

        if k is not None:
            _included_class_indices = heapq.nlargest(k, range(len(_cardinalities)), _cardinalities.__getitem__)
            _included_classes = [_classes[idx] for idx in _included_class_indices]
            classes_dict = {_class:classes_dict[_class] for _class in _included_classes}

        logging.info('Classes: %s' % classes_dict.keys())

        if is_heat:
            show_heat(experiment, classes_dict, save=save_file)
        else:
            show_points(experiment, classes_dict, manifold_method, cluster_method, save=save_file)


if __name__ == '__main__':
    font = {'size': 8}
    matplotlib.rc('font', **font)
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
