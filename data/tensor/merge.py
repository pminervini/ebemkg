#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

import sys
import cPickle as pickle
import logging


def merge(dataset, lhs_path, rel_path, rhs_path, targets_path=None):
    lhs = pickle.load(open(lhs_path, 'rb'))
    rel = pickle.load(open(rel_path, 'rb'))
    rhs = pickle.load(open(rhs_path, 'rb'))

    if dataset == 'kinships':
        Nent = 130
        Nrel = 26
    elif dataset == 'nations':
        Nent = 182
        Nrel = 57
    elif dataset == 'umls':
        Nent = 184
        Nrel = 49
    else:
        raise ValueError('Unknown dataset: %s' % (dataset))

    specs = {
        'Nent': Nent,
        'Nrel': Nrel
    }

    obj = {
        'lhs': lhs,
        'rel': rel,
        'rhs': rhs,
        'specs': specs
    }

    if targets_path is not None:
        targets = pickle.load(open(targets_path, 'rb'))
        obj['targets'] = targets

    return obj


def serialize(obj, path):
    f = open(path, 'wb')
    pickle.dump(obj, f)
    f.close()


def main(argv):

    for dataset in ['kinships', 'nations', 'umls']:
        for fold in range(10):
            train_pos_lhs_path = '%s/original/%s-train-pos-lhs-fold%d.pkl' % (dataset, dataset, fold)
            train_pos_rel_path = '%s/original/%s-train-pos-rel-fold%d.pkl' % (dataset, dataset, fold)
            train_pos_rhs_path = '%s/original/%s-train-pos-rhs-fold%d.pkl' % (dataset, dataset, fold)

            train_neg_lhs_path = '%s/original/%s-train-neg-lhs-fold%d.pkl' % (dataset, dataset, fold)
            train_neg_rel_path = '%s/original/%s-train-neg-rel-fold%d.pkl' % (dataset, dataset, fold)
            train_neg_rhs_path = '%s/original/%s-train-neg-rhs-fold%d.pkl' % (dataset, dataset, fold)

            valid_lhs_path = '%s/original/%s-valid-lhs-fold%d.pkl' % (dataset, dataset, fold)
            valid_rel_path = '%s/original/%s-valid-rel-fold%d.pkl' % (dataset, dataset, fold)
            valid_rhs_path = '%s/original/%s-valid-rhs-fold%d.pkl' % (dataset, dataset, fold)
            valid_targets_path = '%s/original/%s-valid-targets-fold%d.pkl' % (dataset, dataset, fold)

            test_lhs_path = '%s/original/%s-test-lhs-fold%d.pkl' % (dataset, dataset, fold)
            test_rel_path = '%s/original/%s-test-rel-fold%d.pkl' % (dataset, dataset, fold)
            test_rhs_path = '%s/original/%s-test-rhs-fold%d.pkl' % (dataset, dataset, fold)
            test_targets_path = '%s/original/%s-test-targets-fold%d.pkl' % (dataset, dataset, fold)

            train_pos = merge(dataset, train_pos_lhs_path, train_pos_rel_path, train_pos_rhs_path)
            serialize(train_pos, '%s/%s-train-pos-fold%d.pkl' % (dataset, dataset, fold))

            train_neg = merge(dataset, train_neg_lhs_path, train_neg_rel_path, train_neg_rhs_path)
            serialize(train_neg, '%s/%s-train-neg-fold%d.pkl' % (dataset, dataset, fold))

            valid = merge(dataset, valid_lhs_path, valid_rel_path, valid_rhs_path, targets_path=valid_targets_path)
            serialize(valid, '%s/%s-valid-fold%d.pkl' % (dataset, dataset, fold))

            test = merge(dataset, test_lhs_path, test_rel_path, test_rhs_path, targets_path=test_targets_path)
            serialize(test, '%s/%s-test-fold%d.pkl' % (dataset, dataset, fold))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
