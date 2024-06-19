import math
import scipy
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn import model_selection
from sklearn.base import is_classifier
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import get_scorer, get_scorer_names
from .utils import _sample

class Crosser():
    """
    An object for repeated cross-fitting

    Parameters
    ----------
    estimator : estimator object
        The base estimator for repeated cross-fitting. 

    cv : cross-validation generator
        The cross-validation generator for repeated cross-fitting. 
        It must be a class member of `KFold`, `RepeatedKFold`, 
        `ShuffleSplit`, `StratifiedShuffleSplit`.
        
    scoring : str or callable, default=None
        Strategy to evaluate the performance of the cross-validated 
        model when using `summarize`. To see availble scoring methods 
        via `str`, use `sklearn.get_scorer_names()`. 
        
            - For regression tasks, `None` means "r2". 
            - For classification tasks, `None` means "accuracy".

    """

    def __init__(
        self,
        estimator,
        cv,
        scoring = None
    ):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        
        self._check()
        self._setup()
        
    def fit(
        self, 
        X, 
        y
    ):
        """
        Fit by repeated cross-fitting
        
        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Training array, where `n_samples` is the sample size and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target relative to X for classification or regression.

        Returns
        -------
        self : object
            Instance of fitted crosser.
        """    
        
        cv = self.cv
        estimators = []
        features = self._features(X, y, on = "train")
        targets = self._targets(X, y, on = "train")
        for feature, target in zip(features, targets):     
            estimator = clone(self.estimator)
            _ = estimator.fit(feature, target)
            estimators.append(estimator)
        self.estimators_ = estimators
        
        if is_classifier(self):
            label_binarizer = LabelBinarizer()
            _ = label_binarizer.fit(y)
            self.label_binarizer_ = label_binarizer
            self.classes_ = sorted(set(y))
        self._is_fitted = True
            
        val_scores, train_scores, test_scores = self._scores(X, y)
        self.val_scores_ = val_scores
        self.train_scores_ = train_scores
        self.test_scores_ = test_scores
    
        return self

    
    def summarize(
        self,
        cross_fit = None,
        combine = None,
        reverse = False,
        verbose = True
    ):
        """
        Summarize repeated cross-fitting results
        
        Parameters
        ----------

        cross_fit : bool, default=None
            If `True`, score values will be averaged across folds. 
            It is applicable if `n_folds` > 1. If applicable, `None` 
            means `True`; Otherwise, `None` means `False`. 
            
        combine : bool, default=None
            If `True`, score values will be averaged across folds 
            and repeats. It is applicable if `n_splits` > 1. If 
            `n_repeats` > 1, `None` means `True`; Otherwise, `None` 
            means `False`.

        reverse : bool, default=True
            If `True`, negative score values will be reported. 
            Note that by default a larger score value means better 
            in `scikit-learn`.
        
        verbose : bool, default=True
            Controls the verbosity.

        Returns
        -------
        summary : DataFrame
            A summary for validation, train, and test scores (`val_score`, 
            `train_score`, and `test_score`). The validation scores 
            are only presented if `estimator.best_score_` is available. 
            
        """
        check_is_fitted(self)
        estimator = self.estimator
        cv = self.cv
        scoring = self.scoring
        n_splits = self._n_splits
        n_repeats = self._n_repeats
        n_folds = self._n_folds
        
        val_scores = self.val_scores_
        train_scores = self.train_scores_
        test_scores = self.test_scores_
        
        if combine is None:
            if n_repeats > 1:
                combine = True
            else:
                combine = False           
                
        if cross_fit is None:
            if n_folds > 1:
                cross_fit = True
            else:
                cross_fit = False

        if reverse is True:
            val_scores, train_scores, test_scores = -val_scores, -train_scores, -test_scores
            
        if cross_fit:
            val_scores = val_scores.reshape((n_repeats, n_folds)).mean(axis = 1)
            train_scores = train_scores.reshape((n_repeats, n_folds)).mean(axis = 1)
            test_scores = test_scores.reshape((n_repeats, n_folds)).mean(axis = 1)
        
        if not combine:
            if cross_fit:
                index = pd.Index(
                    range(n_repeats), 
                    name = "repeat")
            else:
                splits = list(range(self._n_splits))
                repeats = [split // n_folds for split in splits]
                folds = [split % n_folds for split in splits]
                index = pd.MultiIndex.from_tuples(
                    list(zip(splits, repeats, folds)),
                    names = ["split", "repeat", "fold"])
            summary = pd.DataFrame(
                {"val_score": val_scores,
                 "train_score": train_scores,
                 "test_score": test_scores},
                index = index
            )
        else:
            index = pd.Index(
                    ["mean", "std"], 
                    name = "method")
            summary = pd.DataFrame(
                {"val_score": [np.mean(val_scores), np.std(val_scores)],
                 "train_score": [np.mean(train_scores), np.std(train_scores)],
                 "test_score": [np.mean(test_scores), np.std(test_scores)]},
                index = index
            ) 
            
        if verbose:     
            print("Crosser Summary", end = " ")
            print("(cross_fit=", cross_fit, ", " ,
                  "combine=", combine, ")", sep = "")     
            print(" + Estimator: ", 
                  estimator.__class__.__name__, sep = "")
            print(" + Cross-Validator: ", 
                  cv.__class__.__name__, 
                  " (", "n_folds=", n_folds, 
                  ", n_repeats=", n_repeats, ")", sep = "")
            if callable(scoring):
                print(" + Train/Test Scorer: ", 
                      "Callable", 
                      " (", "reverse=", reverse, ")", sep = "")
            else:
                print(" + Train/Test Scorer: ", 
                      scoring.replace("_", " ").title(), 
                      " (", "reverse=", reverse, ")", sep = "")
        return summary

    
    def predict(
        self, 
        X,
        *,
        split = None
    ):
        check_is_fitted(self)
        if split is None:
            preds = np.array(
                [estimator.predict(X) 
                 for estimator in self.estimators_])
            if is_classifier(self):
                pred = np.apply_along_axis(
                    lambda x: np.argmax(np.bincount(x)),
                    axis = 0,
                    arr = preds)
            else:
                pred = preds.mean(axis = 0)
        else:
            pred = self.estimators_[split].predict(X)
        return pred

    
    def predict_proba(
        self, 
        X,
        *,
        split = None
    ):
        check_is_fitted(self)
        if split is None:
            preds = np.array(
                [estimator.predict_proba(X)
                 for estimator in self.estimators_])
            pred = preds.mean(axis = 0)
        else:
            pred = self.estimators_[split].predict_proba(X)
        return pred

    
    def predict_log_proba(
        self, 
        X,
        *,
        split = None
    ):
        check_is_fitted(self)
        if split is None:
            preds = np.array(
                [estimator.predict_log_proba(X)
                 for estimator in self.estimators_])
            pred = preds.mean(axis = 0)
        else:
            pred = self.estimators_[split].predict_log_proba(X)
        return pred

    
    def decision_function(
        self, 
        X,
        *,
        split = None
    ):
        check_is_fitted(self)
        if split is None:
            preds = np.array(
                [estimator.decision_function(X) 
                 for estimator in self.estimators_])
            pred = preds.mean(axis = 0)
        else:
            pred = self.estimators_[split].decision_function(X)
        return pred
            
    def _scores(
        self,
        X,
        y
    ):
        estimators = self.estimators_
        scorer = self._scorer
        n_repeats = self._n_repeats
        n_folds = self._n_folds

        features = zip(self._features(X, y, on = "train"),
                       self._features(X, y, on = "test"))
        targets = zip(self._targets(X, y, on = "train"),
                      self._targets(X, y, on = "test"))
        
        train_scores = []
        val_scores = []
        test_scores = []
        for estimator, feature, target in zip(estimators, features, targets):
            feature_train, feature_test = feature
            target_train, target_test = target
            train_score = scorer(estimator, feature_train, target_train)
            test_score = scorer(estimator, feature_test, target_test)
            if hasattr(estimator, "best_score_"):
                val_score = estimator.best_score_
            else:
                val_score = np.nan        
            val_scores.append(val_score)
            train_scores.append(train_score)
            test_scores.append(test_score)
        val_scores = np.array(val_scores)
        train_scores = np.array(train_scores)
        test_scores = np.array(test_scores)
        return val_scores, train_scores, test_scores
                   
    
    def _features(
        self,
        X,
        y,
        on = "test"
    ):
        cv = self.cv
        features = []
        for train_index, test_index in cv.split(X, y): 
            if on == "test":
                index = test_index
            else:
                index = train_index
            if isinstance(X, pd.DataFrame):
                feature = X.iloc[index, :]
            else:
                feature = X[index, :]
            features.append(feature)
        return features

    def _targets(
        self,
        X,
        y,
        binarize = None,
        on = "test"
    ):
        cv = self.cv
        targets = []
        for train_index, test_index in cv.split(X, y): 
            if on == "test":
                index = test_index
            else:
                index = train_index
            if isinstance(y, pd.Series):
                target = y.iloc[index]
            else:
                target = y[index]
            if is_classifier(self.estimator):
                if binarize is True:
                    target = self.label_binarizer_.transform(target)
                    if target.shape[1] == 1:
                        target = np.append(1 - target, target, axis=1)    
            targets.append(target)
        return targets

    
    def _rvs(
        self,
        X, 
        y,
        n_copies = None,
        random_state = None
    ):
        estimators = self.estimators_
        features = self._features(X, y)
        targets = self._targets(X, y)
        rvs = []
        for estimator, feature, target in zip(estimators, features, targets):
            if hasattr(estimator, "sample"):
                rv = estimator.sample(
                    feature, 
                    target, 
                    n_copies = n_copies,
                    random_state = random_state)
            else:
                rv = _sample(
                    estimator, 
                    feature, 
                    target, 
                    n_copies = n_copies,
                    random_state = random_state)
            rvs.append(rv)
        return rvs

    def _check(
        self
    ):
        cv = self.cv
        scoring = self.scoring
        
        allowed_cvs = {"KFold", "RepeatedKFold", "ShuffleSplit"}
        if not isinstance(
            cv,
            tuple(getattr(model_selection, allowed_cv) 
                  for allowed_cv in allowed_cvs)
        ):
            raise ValueError(
                "Support `cv` types are {}.".format(allowed_cvs))
        else:
            if isinstance(
                cv, getattr(model_selection, "KFold")):
                if cv.shuffle:
                    if cv.random_state is None:
                        raise ValueError(
                            "When `cv` is `KFold` and `cv.shuffle` is `True`, "
                            "`cv.random_state` cannot be `None`.")                    
            else:
                if cv.random_state is None:
                    raise ValueError(
                        "When `cv` is `RepeatedKFold` or `ShuffleSplit`, "
                        "`cv.random_state` cannot be `None`.")
                
        if scoring is not None:
            if not (isinstance(scoring, str) or callable(scoring)):
                raise ValueError(
                    "Support `scoring` types are `str`, `callable`, or `None`")
            if isinstance(scoring, str):
                scorer_names = get_scorer_names()
                if not scoring in get_scorer_names():
                    raise ValueError(
                        "Support `scoring` names are {}.".format(scorer_names))


    def _setup(
        self
    ):
        estimator = self.estimator
        cv = self.cv
        scoring = self.scoring
        
        kf_cvs = {"KFold", "RepeatedKFold"}
        ss_cvs = {"ShuffleSplit"}
        n_splits = cv.get_n_splits()
        if isinstance(
            cv, 
            tuple(getattr(model_selection, kf_cv) 
                  for kf_cv in kf_cvs)):
            if hasattr(cv, "n_repeats"):
                n_repeats = cv.n_repeats
            else:
                n_repeats = 1
        else:
            n_repeats = cv.get_n_splits()
        n_folds = n_splits // n_repeats

        if scoring is None:
            if is_classifier(estimator):
                scoring = "accuracy"
            else:
                scoring = "r2"
        if isinstance(scoring, str):
            scorer = get_scorer(scoring)
        elif callable(scoring):
            scorer = scoring
        
        self.scoring = scoring
        self._is_fitted = False
        self._estimator_type = estimator._estimator_type
        self._n_splits = n_splits
        self._n_repeats = n_repeats
        self._n_folds = n_folds
        self._scorer = scorer

    def __sklearn_is_fitted__(self):
        return hasattr(self, "_is_fitted") and self._is_fitted
        



