from sklearn.base import clone

def augmentation(model, x,y, iterative: bool = False):
    mod = clone(model)
    mod.fit(x,y)
    weights = 1 / mod.predict_proba(x)[:,0]
    model.fit(x,y,weights)