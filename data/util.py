import numpy as np
import scipy
import scipy.sparse as sp
import theano
import socket
import copy, pickle, codecs

from ConfigParser import SafeConfigParser


def configuration(file='config.ini', encoding='utf-8'):
    parser = SafeConfigParser()
    with codecs.open(file, 'r', encoding=encoding) as f:
        parser.readfp(f)
    return parser

def producer(c):
    producer = {
        'system': c.get('System', 'name'),
        'version': c.get('System', 'version'),
        'host': socket.gethostname()
    }

def convert2idx(spmat):
    rows, cols = spmat.nonzero()
    return rows[np.argsort(cols)]

# Switching from lil_matrix to csc_matrix speeds up column shuffling and sampling A LOT
def tocsr(M):
    return M.tocsr().astype(theano.config.floatX)

class ExpDataSet(object):

    def __init__(self, train_path='data/aifb/standard_noaff/aifb_noaff.pkl', valid_path=None, test_path=None):
        self.train_path = train_path
        self.train_dict = pickle.load(open(train_path))

        # Training
        self.has_train = True
        self.train_lhs = tocsr(self.train_dict['lhs'])
        self.train_rhs = tocsr(self.train_dict['rhs'])
        self.train_rel = tocsr(self.train_dict['rel'])

        self.specs = None
        if 'specs' in self.train_dict.keys():
            self.specs = self.train_dict['specs']

        self.valid_lhs = None
        self.valid_rhs = None
        self.valid_rel = None

        self.has_valid = False
        if valid_path is not None:
            self.valid_path = valid_path
            self.valid_dict = pickle.load(open(valid_path))

            self.has_valid = True

            self.valid_lhs = tocsr(self.valid_dict['lhs'])
            self.valid_rhs = tocsr(self.valid_dict['rhs'])
            self.valid_rel = tocsr(self.valid_dict['rel'])

        self.test_lhs = None
        self.test_rhs = None
        self.test_rel = None

        self.has_test = False
        if test_path is not None:
            self.test_path = test_path
            self.test_dict = pickle.load(open(test_path))

            self.has_test = True

            self.test_lhs = tocsr(self.test_dict['lhs'])
            self.test_rhs = tocsr(self.test_dict['rhs'])
            self.test_rel = tocsr(self.test_dict['rel'])

        self.entities = self.train_dict['entities']
        self.predicates = self.train_dict['predicates']

        self.resources, self.bnodes, self.literals = None, None, None

        if ('resources' in self.train_dict.keys()):
            self.resources = self.train_dict['resources']

        if ('bnodes' in self.train_dict.keys()):
            self.bnodes = self.train_dict['bnodes']

        if ('literals' in self.train_dict.keys()):
            self.literals = self.train_dict['literals']

    # Training set
    def train(self):
        # Positives
        return self.train_lhs, self.train_rhs, self.train_rel

    # Validation set
    def valid(self):
        return self.valid_lhs, self.valid_rhs, self.valid_rel

    # Test set
    def test(self):
        return self.test_lhs, self.test_rhs, self.test_rel


class TensorDataSet(object):

    def __init__(self, train_pos_path=None, train_neg_path=None,
                    valid_path=None, test_path=None):
        self.train_pos_path = train_pos_path
        self.train_neg_path = train_neg_path

        # Training
        self.train_pos_dict = pickle.load(open(train_pos_path))

        self.train_pos_lhs = tocsr(self.train_pos_dict['lhs'])
        self.train_pos_rhs = tocsr(self.train_pos_dict['rhs'])
        self.train_pos_rel = tocsr(self.train_pos_dict['rel'])

        self.train_neg_dict = pickle.load(open(train_neg_path))

        self.train_neg_lhs = tocsr(self.train_neg_dict['lhs'])
        self.train_neg_rhs = tocsr(self.train_neg_dict['rhs'])
        self.train_neg_rel = tocsr(self.train_neg_dict['rel'])

        self.specs = None
        if 'specs' in self.train_pos_dict.keys():
            self.specs = self.train_pos_dict['specs']

        self.valid_path = valid_path
        if self.valid_path is not None:
            self.has_valid = True
            self.valid_dict = pickle.load(open(valid_path))

            self.valid_lhs = tocsr(self.valid_dict['lhs'])
            self.valid_rhs = tocsr(self.valid_dict['rhs'])
            self.valid_rel = tocsr(self.valid_dict['rel'])

            self.valid_targets = self.valid_dict['targets']

        self.test_path = test_path
        if self.test_path is not None:
            self.has_test = True
            self.test_dict = pickle.load(open(test_path))

            self.test_lhs = tocsr(self.test_dict['lhs'])
            self.test_rhs = tocsr(self.test_dict['rhs'])
            self.test_rel = tocsr(self.test_dict['rel'])

            self.test_targets = self.test_dict['targets']

    # Training set
    def train_pos(self):
        # Positives
        return self.train_pos_lhs, self.train_pos_rhs, self.train_pos_rel

    def train_neg(self):
        # Negatives
        return self.train_neg_lhs, self.train_neg_rhs, self.train_neg_rel

    # Validation set
    def valid(self):
        return self.valid_lhs, self.valid_rhs, self.valid_rel, self.valid_targets

    # Test set
    def test(self):
        return self.test_lhs, self.test_rhs, self.test_rel, self.test_targets


def dump_labels(path, name):
    fd = open(path)
    obj = pickle.load(fd)
    tensor = obj['tensor']
    resources, predicates, attributes = [], [], []
    if name == 'umls':
        resources = obj['entity_names']
        predicates = obj['relation_names']
    elif name == 'kinships':
        NR, NP = tensor.shape[0], tensor.shape[2]
        resources = ['resource_' + str(i) for i in range(NR)]
        predicates = ['predicate_' + str(i) for i in range(NP)]
    elif name == 'nations':
        NR, NP = 14, tensor.shape[2]
        NA = tensor.shape[0] - NR
        resources = obj['attname'][0:NR]
        attributes = obj['attname'][NR:]
        predicates = obj['relname']
    return resources, predicates, attributes

# Utils
def create_random_mat(shape, listidx=None):
    """
    This function creates a random sparse index matrix with a given shape. It
    is useful to create negative triplets.

    :param shape: shape of the desired sparse matrix.
    :param listidx: list of index to sample from (default None: it samples from
                    all shape[0] indexes).

    :note: if shape[1] > shape[0], it loops over the shape[0] indexes.
    """
    if listidx is None:
        listidx = np.arange(shape[0])
    listidx = listidx[np.random.permutation(len(listidx))]
    cooData = np.ones(shape[1], dtype=theano.config.floatX)
    cooRowIdxs = listidx[np.arange(shape[1]) % len(listidx)]
    cooColIdxs = range(shape[1])
    randommat = scipy.sparse.coo_matrix((cooData, (cooRowIdxs, cooColIdxs)), shape=shape)
    return randommat.tocsr()

class DD(dict):
    """This class is only used to replace a state variable of Jobman"""

    def __getattr__(self, attr):
        if attr == '__getstate__':
            return super(DD, self).__getstate__
        elif attr == '__setstate__':
            return super(DD, self).__setstate__
        elif attr == '__slots__':
            return super(DD, self).__slots__
        return self[attr]

    def __setattr__(self, attr, value):
        assert attr not in ('__getstate__', '__setstate__', '__slots__')
        self[attr] = value

    def __str__(self):
        return 'DD%s' % dict(self)

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memo):
        z = DD()
        for k, kv in self.iteritems():
            z[k] = copy.deepcopy(kv, memo)
        return z

def load_file(path):
    return sp.csr_matrix(pickle.load(open(path)), dtype=theano.config.floatX)
