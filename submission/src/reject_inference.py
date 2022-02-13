import numpy as np
import sklearn.mixture
from sklearn.base import clone

class augmentation():
    def __init__(self, iterative: bool = False):
        self.iterative = iterative

    def fit(self, model, x,y):
        mod = clone(model)
        mod.fit(x, y)
        prev_pred = mod.predict_proba(x)[:,0]
        improv = 1
        i = 0
        if(self.iterative):
            while(improv > 0.0001 and i < 1000):
                weights = np.divide(1,prev_pred,out=np.ones_like(prev_pred)/0.000001,where=prev_pred!=0)
                model.fit(x, y, weights)
                pred = model.predict_proba(x)[:,0]
                improv = sum((prev_pred - pred)**2)
                prev_pred = pred
                i =+ 1
        else:
            weights = np.divide(1,prev_pred,out=np.ones_like(prev_pred)/0.000001,where=prev_pred!=0)
            model.fit(x, y, weights)

def  EMsemisupervised(model, X, y, accepted):
    params = learn_params(X, accepted)
    
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
    model.fit(X, y, predicts)

def learn_params(x_labeled, y_labeled):
    n = x_labeled.shape[0]
    phi = x_labeled[y_labeled].shape[0] / n
    mu0 = np.sum(x_labeled[y_labeled], axis=0) / x_labeled[y_labeled == 0].shape[0]
    mu1 = np.sum(x_labeled[y_labeled], axis=0) / x_labeled[y_labeled == 1].shape[0]
    sigma0 = np.cov(x_labeled[y_labeled].T, bias= True)
    sigma1 = np.cov(x_labeled[y_labeled].T, bias=True)
    return {'phi': phi, 'mu0': mu0, 'mu1': mu1, 'sigma0': sigma0, 'sigma1': sigma1}