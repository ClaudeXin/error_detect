from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


model = Pipeline([('poly', PolynomialFeatures(degree=2)), 
    ('linear', LinearRegression(fit_intercept=False))])


def train(X, y):
    model.fit(X, y)
    return model
