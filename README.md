# SME++

Prerequisites:

    # apt-get install build-essential git python-dev python-setuptools libopenblas-dev libatlas-base-dev

    # pip install --upgrade numpy scipy git+git://github.com/Theano/Theano.git scikit-learn pymongo tsne  pandas statsmodels patsy seaborn termcolor

    # update-alternatives --config libblas.so
    There are 3 choices for the alternative libblas.so (providing /usr/lib/libblas.so).

    Selection    Path                                   Priority   Status
    ------------------------------------------------------------
    * 0            /usr/lib/openblas-base/libopenblas.so   40        auto mode
    1            /usr/lib/atlas-base/atlas/libblas.so    35        manual mode
    2            /usr/lib/libblas/libblas.so             10        manual mode
    3            /usr/lib/openblas-base/libopenblas.so   40        manual mode

    Press enter to keep the current choice[*], or type selection number: 1


Sample usage on the FreeBase Knowledge Base:

    $ ./learn_parameters.py --strategy=SGD --lr=0.001 --ndim=10 --name=fb --nbatches=1 --train=data/fb/FB-train.pkl --valid=data/fb/FB-valid.pkl --test=data/fb/FB-test.pkl --op=SME_bil
