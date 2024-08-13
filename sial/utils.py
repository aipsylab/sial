import numpy as np
from sklearn.base import is_classifier
from sklearn.preprocessing import LabelBinarizer


def _sample(
    estimator, 
    X, 
    y, 
    n_copies = None,
    random_state = None):
    rng = np.random.default_rng(random_state)
    if estimator is None:
        if n_copies is None:
            rv = rng.permutation(y)
        else:
            rv = np.array(
                [rng.permutation(y) for copy in range(n_copies)])
    else:
        if is_classifier(estimator):
            pred = estimator.predict_proba(X)
            label_binarizer = LabelBinarizer()
            _ = label_binarizer.fit(estimator.classes_)
            if n_copies is None:            
                rv = rng.multinomial(1, pred)
                rv = label_binarizer.inverse_transform(rv)
            else:
                rv = rng.multinomial(
                    1, pred, (n_copies, len(pred)))
                rv = np.array(
                    [label_binarizer.inverse_transform(rv_i) 
                     for rv_i in rv])
        else:
            pred = estimator.predict(X)
            res = y - pred
            if n_copies is None:
                rv = pred + rng.permutation(res)
            else:
                rv = pred + np.array(
                    [rng.permutation(res) for copy in range(n_copies)])
    return rv


class NotInferredError(ValueError):
    """Exception class to raise if inferer is used before infering.
    """
    