#!/usr/bin/python -uB
# -*- coding: utf-8 -*-


import numpy as np
import theano

import random, pickle, datetime, time
import os, sys, types, socket, getopt, logging

import data.util as util

import energy.sparse.learning as learning

import energy.activation as activation
import energy.similarity as similarity
import energy.loss as loss
import energy.evaluation as evaluation

import energy.model as model

import persistence.layer as persistence


# Experiment function
#@profile
def learn(state):
    np.random.seed(state.seed)

    c = util.configuration()
    layer, exp, exp_id = None, {}, None

    dataset = util.TensorDataSet(train_pos_path=state.train_pos_path, train_neg_path=state.train_neg_path,
                                    valid_path=state.valid_path, test_path=state.test_path)

    # Training set (positives)
    trainl, trainr, traino = dataset.train_pos()
    # Training set (negatives)
    trainln, trainrn, trainon = dataset.train_neg()

    # Validation set
    if dataset.has_valid is True:
        validl, validr, valido, valid_targets = dataset.valid()

    # Test set
    if dataset.has_test is True:
        testl, testr, testo, test_targets = dataset.test()

    logging.info('Shape for training set: %s' % (str(trainl.shape)))

    if state.use_db:
        is_fast = False
        if trainl.shape[1] > 10000000: # if the dataset is not small-sized (> 10m triples), switch to fast mode
            is_fast = True

        layer = persistence.PickleLayer(dir=c.get('Persistence', 'path'), is_fast=is_fast)
        exp = {
            'start_time': datetime.datetime.utcnow()
        }
        exp_id = layer.create(state.name, exp)

    exp['best'] = {} # use the validation set (if available) to pick the best model

    state_str = { k: (state[k].__name__ if isinstance(state[k], types.FunctionType) else str(state[k])) for k in state.keys() }
    if 'use_db' in state_str:
        del state_str['use_db']

    exp['state'] = state_str

    exp['producer'] = {
        'system': c.get('System', 'name'),
        'version': c.get('System', 'version'),
        'host': socket.gethostname()
    }

    state.Nrel = dataset.specs['Nrel']
    state.Nent = dataset.specs['Nent']

    if state.op == 'SE' or state.op == 'TransE' or state.op == 'TransH' or state.op == 'ScalE':
        traino = traino[-state.Nrel:, :] # last elements of traino
        trainon = trainon[-state.Nrel:, :]

        if dataset.has_valid is True:
            valido = valido[-state.Nrel:, :]

        if dataset.has_test is True:
            testo = testo[-state.Nrel:, :]

    # Operators
    leftop, rightop = model.op(state.op, state.ndim, state.nhid)

    logging.debug('Initializing the embeddings ..')

    # Embeddings
    embeddings = learning.Embeddings(np.random, state.Nent, state.ndim, tag='emb')

    if state.op == 'SE' and type(embeddings) is not list:
        relationl = learning.Embeddings(np.random, state.Nrel, state.ndim * state.nhid, tag='rell')
        relationr = learning.Embeddings(np.random, state.Nrel, state.ndim * state.nhid, tag='relr')
        embeddings = [embeddings, relationl, relationr]

    if (state.op == 'TransE' or state.op == 'ScalE') and type(embeddings) is not list:
        relationVec = learning.Embeddings(np.random, state.Nrel, state.ndim, tag='relvec')
        embeddings = [embeddings, relationVec, relationVec]

    if state.op == 'TransH' and type(embeddings) is not list:
        projectionVec = learning.Embeddings(np.random, state.Nrel, state.ndim, tag='projvec')
        relationVec = learning.Embeddings(np.random, state.Nrel, state.ndim, tag='relvec')
        embeddings = [embeddings, projectionVec, relationVec] # x, w, d

    simfn = state.simfn

    logging.debug('Initializing the training function ..')

    # Functions compilation
    trainfunc = learning.TrainFn(simfn, embeddings, leftop, rightop,
                                        method=state.method, op=state.op, loss=loss.hinge, loss_margin=state.loss_margin,
                                        decay=state.decay, epsilon=state.epsilon, max_learning_rate=state.max_lr, transh_c=state.C,
                                        weight_L1_param_regularizer=state.l1_param_weight, weight_L2_param_regularizer=state.l2_param_weight)

    testfunc = learning.SimFn(simfn, embeddings, leftop, rightop, op=state.op)

    out, outb = [], []
    valid_aucprs, test_aucprs = [], []

    state.besttrain, state.bestvalid, state.besttest = None, None, None
    state.bestepoch = None

    batchsize = trainl.shape[1] / state.nbatches

    logging.info("Starting the Experiment ..")
    timeref = time.time()

    average_costs_per_epoch = []                                        # X
    ratios_violating_examples_per_epoch = []                            # X

    for epoch_count in range(1, state.totepochs + 1):

        logging.debug('Running epoch %d of %d ..' % (epoch_count, state.totepochs))

        # Shuffling
        order = np.random.permutation(trainl.shape[1])
        trainl, trainr, traino = (trainl[:, order], trainr[:, order], traino[:, order])

        order = np.random.permutation(trainln.shape[1])
        trainln, trainrn, trainon = (trainln[:, order], trainrn[:, order], trainon[:, order])

        epoch_average_costs = []                                        # X
        epoch_ratios_violating_examples = []                            # X

        for i in range(state.nbatches): # Iterate over Batches

            logging.debug('Running on batch %d of %d ..' % (i, state.nbatches))

            tmpl = trainl[:, i * batchsize:(i + 1) * batchsize]
            tmpr = trainr[:, i * batchsize:(i + 1) * batchsize]
            tmpo = traino[:, i * batchsize:(i + 1) * batchsize]

            tmpln = trainln[:, i * batchsize:(i + 1) * batchsize]
            tmprn = trainrn[:, i * batchsize:(i + 1) * batchsize]
            tmpon = trainon[:, i * batchsize:(i + 1) * batchsize]

            logging.debug('Executing the training function ..')

            # training iteration
            _lrparam = state.lrparam / float(batchsize)
            outtmp = trainfunc(state.lremb, _lrparam, tmpl, tmpr, tmpo, tmpln, tmprn, tmpon)

            out += [outtmp[0] / float(batchsize)]
            outb += [outtmp[1]]

            average_cost = outtmp[0]                                        # X
            ratio_violating_examples = outtmp[1]                            # X

            epoch_average_costs += [average_cost]                           # X
            epoch_ratios_violating_examples += [ratio_violating_examples]   # X

            logging.debug('Normalizing the embeddings ..')

            # embeddings normalization
            if type(embeddings) is list:
                if state.op == 'TransH':
                    embeddings[1].normalize() # normalize w
                else:
                    embeddings[0].normalize() # normalize e
            else:
                embeddings.normalize()

        # End of Epoch
        logging.info("-- EPOCH %s (%s seconds):" % (epoch_count, round(time.time() - timeref, 3)))

        # Model Evaluation
        logging.info("COST >> %s +/- %s, %% updates: %s%%" % (round(np.mean(out), 4), round(np.std(out), 4), round(np.mean(outb) * 100, 3)))

        out, outb = [], []

        # Evaluate the Ranking Score each test_all epochs
        if (state.test_all is not None) and ((epoch_count % state.test_all) == 0):

            # Evaluation on the Validation Set
            if dataset.has_valid is True:
                valsim = testfunc(validl, validr, valido)[0]
                state.valid = evaluation.auc_pr(predictions=valsim, labels=valid_targets)
                valid_aucprs += [state.valid]
                logging.info("\tAUC-PR >> valid: %s" % (state.valid))

            if dataset.has_test is True:
                testsim = testfunc(testl, testr, testo)[0]
                state.test = evaluation.auc_pr(predictions=testsim, labels=test_targets)
                test_aucprs += [state.test]
                logging.info("\tAUC-PR >> test: %s" % (state.test))

        timeref = time.time()
    return


def launch(op='TransE', simfn=similarity.dot, ndim=20, nhid=20, Nsyn=None, loss_margin=1., test_all=1, use_db=False, seed=666,
        method='SGD', lremb=0.01, lrparam=0.1, decay=0.999, epsilon=1e-6, max_lr=None, nbatches=100, totepochs=2000, C=1.0, name='tmp',
        l1_param_weight=None, l2_param_weight=None, train_pos_path=None, train_neg_path=None, valid_path=None, test_path=None):

    # Argument of the experiment script
    state = util.DD()

    state.name = name
    state.train_pos_path = train_pos_path
    state.train_neg_path = train_neg_path

    state.valid_path = valid_path
    state.test_path = test_path

    state.method = method
    state.op = op
    state.simfn = simfn
    state.ndim = ndim
    state.nhid = nhid
    state.Nsyn = Nsyn

    state.loss_margin = loss_margin
    state.test_all = test_all
    state.use_db = use_db

    state.lremb = lremb
    state.lrparam = lrparam
    state.decay = decay
    state.epsilon = epsilon
    state.max_lr = max_lr
    state.C = C

    state.l1_param_weight = l1_param_weight
    state.l2_param_weight = l2_param_weight

    state.nbatches = nbatches
    state.totepochs = totepochs
    state.seed = seed

    learn(state)


def main(argv):

    name, method, op = 'umls', 'SGD', 'TransE'

    dataset_name = 'umls'

    train_pos_path = 'data/tensor/%s/%s-train-pos-fold0.pkl' % (dataset_name, dataset_name)
    train_neg_path = 'data/tensor/%s/%s-train-neg-fold0.pkl' % (dataset_name, dataset_name)
    valid_path = 'data/tensor/%s/%s-valid-fold0.pkl' % (dataset_name, dataset_name)
    test_path = 'data/tensor/%s/%s-test-fold0.pkl' % (dataset_name, dataset_name)

    lr, decay, epsilon = 1.0, 0.999, 1e-6
    lremb, lrparam = lr, lr
    C, max_lr = 1.0, None

    ndim, nhid = 50, 50
    nbatches = 100
    totepochs = 2000

    test_all, use_db = None, False
    simfn = similarity.dot
    seed = 666

    l1_param_weight, l2_param_weight = None, None

    usage_str = ("""Usage: %s [-h]
                    [--name=<name>] [--train_pos=<path>] [--train_neg=<path>] [--valid=<path>] [--test=<path>]
                    [--use_db] [--op=<op>] [--strategy=<strategy>] [--ndim=<ndim>] [--nhid=<nhid>]
                    [--lr=<lr>] [--decay=<decay>] [--epsilon=<epsilon>] [--max_lr=<max_lr>] [--C=<C>]
                    [--l1_param=<weight>] [--l2_param=<weight>]
                    [--nbatches=<nbatches>] [--test_all=<test_all>] [--totepochs=<totepochs>]  [--seed=<seed>]
                    """ % (sys.argv[0]))

    # Parse arguments
    try:
        opts, args = getopt.getopt(argv, 'h', [ 'name=', 'train_pos=', 'train_neg=', 'valid=', 'test=',
                                                'use_db', 'op=', 'strategy=', 'ndim=', 'nhid=',
                                                'lremb=', 'lrparam=', 'lr=', 'decay=', 'epsilon=', 'max_lr=', 'C=',
                                                'l1_param=', 'l2_param=',
                                                'nbatches=', 'test_all=', 'totepochs=', 'seed=' ])
    except getopt.GetoptError:
        logging.warn(usage_str)
        sys.exit(2)

    for opt, arg in opts:

        if opt == '-h':
            logging.info(usage_str)
            logging.info('\t--name=<name> (default: %s)' % (name))
            logging.info('\t--train_pos=<path> (default: %s)' % (train_pos_path))
            logging.info('\t--train_neg=<path> (default: %s)' % (train_neg_path))
            logging.info('\t--valid=<path> (default: %s)' % (valid_path))
            logging.info('\t--valid=<path> (default: %s)' % (test_path))

            logging.info('\t--use_db (use a persistence layer -- default: %s)' % (use_db))
            logging.info('\t--op=<op> (default: %s)' % (op))
            logging.info('\t--strategy=<strategy> (default: %s)' % (method))
            logging.info('\t--ndim=<ndim> (default: %s)' % (ndim))
            logging.info('\t--nhid=<nhid> (default: %s)' % (nhid))

            logging.info('\t--lr=<lr> (default: %s)' % (lr))
            logging.info('\t--decay=<decay> (default: %s)' % (decay))
            logging.info('\t--epsilon=<epsilon> (default: %s)' % (epsilon))
            logging.info('\t--max_lr=<max_lr> (default: %s)' % (max_lr))
            logging.info('\t--C=<C> (default: %s)' % (C))

            logging.info('\t--l1_param=<weight> (default: %s)' % (l1_param_weight))
            logging.info('\t--l2_param=<weight> (default: %s)' % (l2_param_weight))

            logging.info('\t--nbatches=<nbatches> (default: %s)' % (nbatches))
            logging.info('\t--test_all=<test_all> (default: %s)' % (test_all))
            logging.info('\t--totepochs=<totepochs> (default: %s)' % (totepochs))
            logging.info('\t--seed=<seed> (default: %s)' % (seed))

            return

        if opt == '--name':
            name = arg
        if opt == '--train_pos':
            train_pos_path = arg
        if opt == '--train_neg':
            train_neg_path = arg
        if opt == '--valid':
            valid_path = arg
        if opt == '--test':
            test_path = arg

        elif opt == '--use_db':
            use_db = True
        elif opt == '--op':
            op = arg
        elif opt == '--strategy':
            method = arg
        elif opt == '--ndim':
            ndim = int(arg)
        elif opt == '--nhid':
            nhid = int(arg)
        elif opt == '--lr':
            lr = float(arg)
            lremb = float(arg)
            lrparam = float(arg)
        elif opt == '--lremb':
            lremb = float(arg)
        elif opt == '--lrparam':
            lrparam = float(arg)
        elif opt == '--decay':
            decay = float(arg)
        elif opt == '--epsilon':
            epsilon = float(arg)
        elif opt == '--max_lr':
            max_lr = float(arg)
        elif opt == '--C':
            C = float(arg)

        elif opt == '--l1_param':
            l1_param_weight = float(arg)
        elif opt == '--l2_param':
            l2_param_weight = float(arg)

        elif opt == '--nbatches':
            nbatches = int(arg)
        elif opt == '--test_all':
            test_all = int(arg)
        elif opt == '--totepochs':
            totepochs = int(arg)
        elif opt == '--seed':
            seed = int(arg)

    if op == 'TransE' or op == 'TransH':
        # In TransE and TransH, d(x, y) = ||x - y||_1, and ndim = nhid
        simfn = similarity.L1
        nhid = ndim

    launch(op=op, simfn=simfn, method=method, seed=seed, totepochs=totepochs, name=name,
            train_pos_path=train_pos_path, train_neg_path=train_neg_path, valid_path=valid_path, test_path=test_path,
            use_db=use_db, test_all=test_all, ndim=ndim, nhid=nhid, nbatches=nbatches, Nsyn=None,
            lremb=lremb, lrparam=lrparam, epsilon=epsilon, decay=decay, max_lr=max_lr, C=C,
            l1_param_weight=l1_param_weight, l2_param_weight=l2_param_weight)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
