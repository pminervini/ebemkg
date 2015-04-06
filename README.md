# Energy-Based Embedding Models for Knowledge Graphs

Prerequisites:

    # apt-get install build-essential git python-dev python-setuptools libopenblas-dev libatlas-base-dev parallel
    # pip install --upgrade git+git://github.com/Theano/Theano.git
    # pip install --upgrade numpy scipy  scikit-learn pymongo tsne  pandas statsmodels patsy seaborn termcolor

The following commands use GNU Parallel for executing multiple experiments (default: 8) at the same time.

# Evaluating the Energy Functions:

Freebase (FB15k):

    $ ./scripts/fb15k/fb15k.py | parallel -j 8

WordNet:

    $ ./scripts/wn/wn.py | parallel -j 8

Validation and test results will be stored in directories logs/wn and logs/fb15k.

# Comparing the Learning Algorithms:

Freebase (FB15k):

    $ ./scripts/fb15k_optimization/fb15k_optimal.py | parallel -j 8

WordNet:

    $ ./scripts/wn_optimization/wn_optimal.py | parallel -j 8

Visualizing the minimization of the loss functional using various adaptive learning rates:

    $ BEST_K=1 ./show_losses.py models/wn_opt/*.pkl -show
    $ BEST_K=1 LOSS_THR=10000 ./show_losses.py models/fb15k_opt/*.pkl -show

![Visualization](http://slides.neuralnoise.com/plots_wn_fb15k.png)
