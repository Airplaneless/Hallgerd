import time
import argparse
import warnings
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

from hallgerd.classic import LogisticRegressionCL


def getScore(clf, Xt, yt, Xv, yv):
    t1 = time.time()
    clf.fit(Xt, yt)
    if type(clf) == LogisticRegression:
        y_pred = clf.predict_proba(Xv)[:,1]
    else:
        y_pred = clf.predict_proba(Xv)
    score = roc_auc_score(yv, y_pred)
    total_time = time.time() - t1
    return score, total_time


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--nsamples', type=int, help='number of samples dataset')
    parser.add_argument('--nfeatures', type=int, help='number of features')
    args = parser.parse_args()

    if args.nfeatures is None and args.nsamples is None:
        nsamples = 100000
        nfeatures = 200
    elif args.nfeatures is None:
        nsamples = args.nsamples
        nfeatures = 200
    elif args.nsamples is None:
        nsamples = 100000
        nfeatures = args.nfeatures
    else:
        nsamples = args.nsamples
        nfeatures = args.nfeatures
    # Create dataset 
    X, y = make_classification(n_samples=nsamples, n_features=nfeatures)
    y[y == 0] = -1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    # Init models
    clf_cl = LogisticRegressionCL(verbose=False, max_iter=300, C=1.0, tol=1e-42)
    clf_sk = LogisticRegression(verbose=False, max_iter=300, solver='sag', tol=1e-42)
    # Train and evaluate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print('Training and evaluation on dataset:\n', nsamples, ' samples\n', nfeatures, ' features')
        score_cl, time_cl = getScore(clf_cl, X_train, y_train, X_test, y_test)
        print('### LogRegCL, 300 iter ###\n\troc_auc: ', score_cl, '\n\ttime: ', time_cl)
        score_sk, time_sk = getScore(clf_sk, X_train, y_train, X_test, y_test)
        print('### LogReg with SAGA solver, 300 iter ###\n\troc_auc: ', score_sk, '\n\ttime: ', time_sk)
