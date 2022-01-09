import sklearn.linear_model

class logreg_augmented(sklearn.linear_model.LogisticRegression):
    def fit(self,x,y, sample_weight = None):
        model = sklearn.linear_model.LogisticRegression()
        model.fit(x, y)
        weights = 1/model.predict(x)
        return model.fit(x,y,weights)