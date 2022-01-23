import numpy as np
import sklearn.mixture
from sklearn.base import clone

def augmentation(model, x,y, iterative: bool = False):
    mod = clone(model)
    mod.fit(x,y)
    weights = 1 / mod.predict_proba(x)[:,1]
    model.fit(x,y,weights)


def  EMsemisupervised(model, X,y, accepted):
    params = learn_params(X,accepted)
    
    weights = [1 - params["phi"], params["phi"]]
    means = [params["mu0"], params["mu1"]]
    covariances = [params["sigma0"], params["sigma1"]]

    mixture = sklearn.mixture.GaussianMixture(n_components=2,
                            covariance_type='full',
                            tol=0.01,
                            max_iter=1000,
                            weights_init=weights,
                            means_init=means,
                            precisions_init=covariances)
    mixture.fit(X)
    predicts = mixture.predict_proba(X)[:,1]
    model.fit(X,y,predicts)

def learn_params(x_labeled, y_labeled):
    n = x_labeled.shape[0]
    phi = x_labeled[y_labeled].shape[0] / n
    mu0 = np.sum(x_labeled[y_labeled], axis=0) / x_labeled[y_labeled == 0].shape[0]
    mu1 = np.sum(x_labeled[y_labeled], axis=0) / x_labeled[y_labeled == 1].shape[0]
    sigma0 = np.cov(x_labeled[y_labeled].T, bias= True)
    sigma1 = np.cov(x_labeled[y_labeled].T, bias=True)
    return {'phi': phi, 'mu0': mu0, 'mu1': mu1, 'sigma0': sigma0, 'sigma1': sigma1}