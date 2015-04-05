#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

import os
import sys
import cmd

import numpy

import theano
import theano.tensor as T

import fnmatch

import data.util as util
import persistence.layer as persistence
import energy.similarity as similarity

import cPickle as pickle
import getopt
import logging

# Sample usage: ./exploration_shell.py --experiment=sample_models/aifb_adagrad_5_50_141320.pkl

class ExplorationShell(cmd.Cmd):
    """Exploration Shell"""

    def __init__(self, name, version, experiment, names):
        self.name = name
        self.version = version
        self.experiment = experiment
        self.names = names

        self.default_top_k = 5

        cmd.Cmd.__init__(self)

        self.experiment_name = self.experiment['name']
        self.experiment_state = self.experiment['state']

        self.intro  = (("\nWelcome to the %s %s console!\n"
                        "   Experiment name: %s\n"
                        "   Experiment state: %s\n")
                            % (self.name, self.version, self.experiment_name, self.experiment_state))

        self.prompt = ('%s> ' % self.experiment_name)
        self.load(self.experiment)

    def load(self, experiment):
        best = self.experiment['best']

        self.entities, self.predicates = best['entities'], best['predicates']
        self.entities_set = set(self.entities)

        self.entity_position = {entity:position for (position, entity) in enumerate(self.entities)}
        self.predicate_position = {predicate:position for (position, predicate) in enumerate(self.predicates)}

        if self.names is not None:
            for (name, code) in self.names.items():
                if code in self.entities_set:
                    position = self.entity_position[code]
                    self.entity_position[name] = position

        self.parameters = best['parameters']

        self.embeddings = self.parameters['Eemb']
        self.translations = self.parameters['Erelvec']


        # Get embeddings
        Xt = numpy.asarray(self.embeddings['value'])
        # (N, ndim) parameters matrix
        self.X = Xt.T
        self.N, self.ndim = self.X.shape[0], self.X.shape[1]


        # Get translations
        Rt = numpy.asarray(self.translations['value'])
        # (N, ndim) parameters matrix
        self.R = Rt.T
        self.Nr = self.R.shape[0]

    def complete_ls(self, text, line, begidx, endidx):
        mlines = line.split(' ')

        start = mlines[-1]
        ret = [entity for entity in self.entity_position.keys() if entity.startswith(start)]

        return ret

    def do_ls(self, line):
        mlines = line.split(' ')

        if (mlines[0] == ''):
            results = [element for element in self.entity_position.keys()]
        else:
            results = []
            for mline in mlines:
                results += [element for element in self.entity_position.keys() if fnmatch.fnmatch(element, mline)]

        for result in set(results):
            print(result)

    def help_ls(self):
        print('syntax: ls [pattern]')
        print('-- find entities that match a given expression')

    def complete_lsp(self, text, line, begidx, endidx):
        mlines = line.split(' ')

        start = mlines[-1]
        ret = [predicate for predicate in self.predicate_position.keys() if entity.startswith(start)]

        return ret

    def do_lsp(self, line):
        mlines = line.split(' ')

        if (mlines[0] == ''):
            results = [element for element in self.predicate_position.keys()]
        else:
            results = []
            for mline in mlines:
                results += [element for element in self.predicate_position.keys() if fnmatch.fnmatch(element, mline)]

        for result in set(results):
            print(result)

    def help_lsp(self):
        print('syntax: lsp [pattern]')
        print('-- find predicates that match a given expression')

    def complete_neighbors(self, text, line, begidx, endidx):
        mlines = line.split(' ')

        start = mlines[-1]
        ret = [entity for entity in self.entity_position.keys() if entity.startswith(start)]
        return ret

    def do_neighbors(self, line):
        idx, mlines = None, line.split(' ')

        if (mlines[0] == ''):
            self.help_neighbors()
            return
        try:
            #idx = self.entities.index(mlines[0])
            idx = self.entity_position[mlines[0]]
        except ValueError:
            print('Element not found')

        if idx is not None:
            logging.info('Element index: %d' % idx)

            # (ndim, 1) matrix containing the embedding of the selected resource
            x = self.X[idx, :].reshape(self.ndim, 1)
            # (N, 1) ones matrix
            ones = numpy.ones((self.N, 1))

            # (ndim, N) matrix containing N times the (ndim, 1) matrix (broadcasting)
            B = (x.dot(ones.T)).T

            # calculate the similarity between B and X
            M1, M2 = T.matrix('M1'), T.matrix('M2')
            fsim = theano.function([M1, M2], similarity.L1(M1, M2))

            similarity_values = fsim(B, self.X)
            similarity_ranking = numpy.argsort(similarity_values)[::-1]

            top_k = self.default_top_k
            if len(mlines) > 1:
                top_k = int(mlines[1])

            for i in range(top_k):
                pos = similarity_ranking[i]
                results = [entity for (entity, position) in self.entity_position.items() if position == pos]

                results_str = str(results)
                res_short = results_str[:75] + (results_str[75:] and '..')

                print('%d: %s' % (i, res_short))

    def help_neighbors(self):
        print('syntax: neighbors <entity> [N]')
        print('-- find N entities with embeddings similar to the given one (default N: %d)' % (self.default_top_k))

    def complete_query(self, text, line, begidx, endidx):
        mlines = line.split(' ')

        start = mlines[-1]

        if len(mlines) > 1:
            ret = [predicate for predicate in self.predicates if predicate.startswith(start)]
        else:
            ret = [entity for entity in self.entity_position.keys() if entity.startswith(start)]
        return ret

    def do_query(self, line):
        idx, mlines = None, line.split(' ')

        if (mlines[0] == ''):
            self.help_neighbors()
            return

        try:
            idx = self.entity_position[mlines[0]]
        except ValueError:
            print('Element %s not found' % (mlines[0]))

        predicates_idx = []
        for c in range(1, len(mlines)):
            if c == (len(mlines) - 1) and mlines[c].isdigit():
                pass
            else:
                try:
                    predicate_idx = self.predicates.index(mlines[c])
                    predicates_idx += [predicate_idx]
                except ValueError:
                    print('Element %s not found' % (mlines[c]))

        if idx is not None:
            logging.info('Entity index: %d' % idx)

            # (ndim, 1) matrix containing the embedding of the selected resource
            x = self.X[idx, :].reshape(self.ndim, 1)


            # Traverse the embedding space using translations
            for predicate_idx in predicates_idx:
                logging.info('Predicate index: %d' % predicate_idx)
                t = self.R[predicate_idx, :].reshape(self.ndim, 1)

                x = x + t
                # Normalize
                #x = x / numpy.sqrt(numpy.sum(x ** 2, axis=0))


            # (N, 1) ones matrix
            ones = numpy.ones((self.N, 1))

            # (ndim, N) matrix containing N times the (ndim, 1) matrix (broadcasting)
            B = (x.dot(ones.T)).T

            # calculate the similarity between B and X
            M1, M2 = T.matrix('M1'), T.matrix('M2')
            fsim = theano.function([M1, M2], similarity.L1(M1, M2))

            similarity_values = fsim(B, self.X)
            similarity_ranking = numpy.argsort(similarity_values)[::-1]

            top_k = self.default_top_k
            if mlines[-1].isdigit():
                top_k = int(mlines[-1])

            for i in range(top_k):
                pos = similarity_ranking[i]
                results = [entity for (entity, position) in self.entity_position.items() if position == pos]

                results_str = str(results)
                res_short = results_str[:75] + (results_str[75:] and '..')

                print('%d: %s' % (i, res_short))

    def help_query(self):
        print('syntax: query <entity> <predicate> .. <predicate> [N]')
        print('-- find N entities with embeddings similar to the given one, translated using the given predicates (default N: %d)' % (self.default_top_k))

    def do_exit(self, arg):
        sys.exit(1)

    def help_exit(self):
        print('syntax: exit')
        print('-- terminates the application')

    def do_EOF(self, line):
        return True


def main(argv):
    experiment_id = None
    names_pkl = None

    conf = util.configuration()

    usage_str = ('Usage: %s [-h] [--experiment=<id>] [--names=<file.pkl>]' % (sys.argv[0]))
    # Parse arguments
    try:
        opts, args = getopt.getopt(argv, 'h', ['experiment=', 'names='])
    except getopt.GetoptError:
        logging.warn(usage_str)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            logging.info(usage_str)
            logging.info('\t--experiment=<id> (default: %s)' % (experiment_id))
            logging.info('\t--names=<file.pkl> (default: %s)' % (names_pkl))
            return

        if opt == '--experiment':
            experiment_id = arg
        if opt == '--names':
            names_pkl = arg

    name = conf.get('System', 'name')
    version = conf.get('System', 'version')

    # Allocating the persistence layer ..
    layer = persistence.PickleLayer(dir=conf.get('Persistence', 'path'))
    experiment = layer.get(experiment_id)

    names = None

    if names_pkl is not None:
        names = pickle.load(open(names_pkl, 'rb'))
        names = {name[1:-1]:code for (name, code) in names.items()}

    shell = ExplorationShell(name, version, experiment, names)
    shell.cmdloop()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
