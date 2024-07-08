import warnings
import math
import scipy
import warnings
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import is_classifier
from scipy.special import xlogy
from sklearn.utils import check_random_state
from itertools import repeat
from .crosser import Crosser
from .utils import _sample
from .utils import NotInferredError

class Inferer():
    """
    An inferer for making statistical inference

    Parameters
    ----------
    learner : estimator or crosser object 
        The learner being infered. 

    sampler : estimator or crosser object
        A sampler that predicts $x_j$ by $x_{-j}$. When `sampler` is 
        a crosser, its cross-validator (splitter) must be identical 
        to the learner. If `sampler` doesn't have a sample method, 
        the residual permutation approach will be used for resampling.

    competitor : estimator or crosser object
        A competitor that predicts $y$ by $x_{-j}$. If competitor is 
        a crosser, its cross-validator (splitter) must be identical 
        to the learner.

    removal : int, str, or list
        A label for removing $x_j$. 
            - When `learner` is trained with numpy array, `removal` 
              must be column index(es) of $x_j$. 
            - When `learner` is trained with `DataFrame`, `removal` 
              must be column name(s) of $x_j$.  
    
    method : {"HRT", "RPT", "CPI", "LOCO", "PIE"}
        The method for making inference. It provides a control
        for implementing a specific procedure.

    loss_func : {"mean_squared_error", "mean_absolute_error",
                "zero_one_loss", "log_loss"}, default=None
        The loss function for measuring the difference between 
        $y$ and its prediction. 

            - When `learner` is a regressor, `None` means 
              "mean_squared_error".
            - When `learner` is a classifier, `None` means 
              "log_loss".

    null_dist : {"resampling", "normality", "permutation"}, \
                 default=None
        The way for constructing a null distribution of test statistic.
        
            - When `method` is "HRT", "RPT", `None` means "resampling".
            - When `method` is "CPI", "LOCO" or "PIE", `None` means 
              "normality".     

    double_split : bool, default=None
        An indicator for double splitting. If applicable, 
        
            - When `method` is "LOCO", `None` means `False`
            - When `method` is "PIE", `None` means `True`.

    perturb_size : non-negative float, default=None
        The standard deviation of random error for pertubation. 
        If applicable, `None` means 0.

    n_copies : int, default=None
        The number of copies for resampling $x_j$ given $x_{-j}$. 
        If applicable,
        
            - When `method` is "HRT", "RPT", `None` means 2000.
            - When `method` is "CPI", `None` means 1.
              
    n_permutations : int, default=None
        The number of permutations. If applicable, `None` means 2000.

    random_state : int or None, default=None
        Controls the randomness of sampling and permutation. Unlike
        `random_state` in scikit-learn, `RandomState` is not allowed 
        here.
    """
    
    def __init__(
        self, 
        learner,
        sampler = None,
        competitor = None,
        removal = None,
        method = None,
        loss_func = None,
        null_dist = None,
        double_split = None,
        perturb_size = None,
        n_copies = None,
        n_permutations = None,
        random_state = None):

        self.learner = learner
        self.sampler = sampler
        self.competitor = competitor
        self.removal = removal
        self.method = method
        self.loss_func = loss_func
        self.null_dist = null_dist
        self.double_split = double_split
        self.perturb_size = perturb_size
        self.n_copies = n_copies
        self.n_permutations = n_permutations
        self.random_state = random_state

    def summarize(
        self,
        cross_fit = None,
        combine = None,
        reverse = False,
        verbose = True
    ):
        """
        Summarize inference results
        
        Parameters
        ----------

        cross_fit : bool, default=None
            If `True`, inference results will be integrated across folds 
            by cross-fitting strategy. It is applicable only if `learner` 
            is a `Crosser` with `n_folds` > 1. If applicable, `None` means 
            `True`; Otherwise, `None` means `False`.

        combine : bool, default=None
            If `True`, inference results will be combined across 
            folds and repeats. For `estimates` and `std_error`, simple 
            average is used for combination. For `p_value`, several 
            p-value combination methods are used. It is applicable only 
            if `learner` is a `Crosser` with `n_splits` > 1. If `n_repeats` 
            > 1, `None` means `True`; Otherwise, `None` means `False`.

        reverse : bool, default=False
            If `True`, negative values of `estimate` will be reported. 

        verbose : bool, default=True
            Controls the verbosity.

        Returns
        -------
        summary : DataFrame
            A summary for inference results.
            
        """
        removal = self.removal
        method = self.method
        loss_func = self.loss_func
        null_dist = self.null_dist
        double_split = self.double_split
        perturb_size = self.perturb_size
        n_copies = self.n_copies
        n_permutations = self.n_permutations
        is_inferred = self._is_inferred
        is_crosser = self._is_crosser
        n_splits = self._n_splits
        n_repeats = self._n_repeats
        n_folds = self._n_folds
        null_dist = self.null_dist
        null_values = self.null_values_

        if not is_inferred:
            raise NotInferredError("This inferer is not inferred yet.")
        
        if combine is None:
            if is_crosser and (n_repeats > 1):
                combine = True
            else:
                combine = False           
                
        if cross_fit is None:
            if is_crosser and (n_folds > 1):
                cross_fit = True
            else:
                cross_fit = False

        sizes, estimates, std_errors, p_values = self._stats()

        if cross_fit:
            sizes = sizes.reshape(
                (n_repeats, n_folds)).sum(axis = 1)
            estimates = estimates.reshape(
                (n_repeats, n_folds)).mean(axis = 1)
            std_errors = (std_errors.reshape(
                (n_repeats, n_folds)).mean(axis = 1) * 
                          np.sqrt(1 / n_folds))
            if (null_dist == "permutation") or (null_dist == "resampling"):
                null_values = np.array(null_values)
                null_values = null_values.reshape(
                    (n_repeats, n_folds, -1)).mean(axis = 1)
                p_values = np.array(
                    [(null_value > estimate).mean() 
                     for estimate, null_value in zip(estimates, null_values)])
            else:
                p_values = np.array(
                    [1 - scipy.stats.norm.cdf(estimate / std_error) 
                     for estimate, std_error in zip(estimates, std_errors)])

        if reverse is True:
            estimates = - estimates

        if isinstance(removal, list):
            removal = " + ".join(
                [str(removal_i) for removal_i in removal])
        
        if (not is_crosser) or (len(p_values) == 1):
            index = pd.Index(
                [removal], 
                name = "removal")
            summary = pd.DataFrame(
                    {"size": sizes,
                     "estimate": estimates,
                     "std_error": std_errors,
                     "p_value": p_values},
                    index = index)
        else:    
            if not combine:
                if cross_fit:
                    index = pd.MultiIndex.from_tuples(
                        list(zip(repeat(removal), range(n_repeats))),
                        names = ["removal", "repeat"])
                else:
                    splits = list(range(self._n_splits))
                    repeats = [split // n_folds for split in splits]
                    folds = [split % n_folds for split in splits]
                    index = pd.MultiIndex.from_tuples(
                        list(zip(repeat(removal), splits, repeats, folds)),
                        names = ["removal", "split", "repeat", "fold"])
                summary = pd.DataFrame(
                    {"size": sizes,
                     "estimate": estimates,
                     "std_error": std_errors,
                     "p_value": p_values},
                    index = index)
            else:
                combined_p_values = self._combined_p_values(p_values)
                index = pd.MultiIndex.from_tuples(
                    list(zip(repeat(removal), combined_p_values.keys())),
                    names = ["removal", "method"])
                summary = pd.DataFrame(
                    {"size": [np.mean(sizes)] * len(combined_p_values),
                     "estimate": [np.mean(estimates)] * len(combined_p_values),
                     "std_error": [np.mean(std_errors)] * len(combined_p_values),
                     "p_value": combined_p_values.values()},
                    index = index)
        if verbose:
            print("Inferer Summary", end = " ")
            print("(cross_fit=", cross_fit, ", " ,
                  "combine=", combine, ")", sep = "")     
            print(" + Method:", method, end = " ")
            print("(double_split=", double_split, ", " ,
                  "perturb_size=", perturb_size, ")", sep = "")    
            print(" + Null Distribution:", null_dist.title(), end = " ")
            print("(n_copies=", n_copies, ", " ,
                  "n_permutations=", n_permutations, ")", sep = "")
            print(" + Loss Function: ", 
                  loss_func.__name__.replace("_", " ").title(),
                 " (", "reverse=", reverse, ")", sep = "")
        return summary

    def _stats(
        self
    ):
        null_dist = self.null_dist
        double_split = self.double_split
        n_repeats = self._n_repeats
        n_folds = self._n_folds
        l_losses = self.l_losses_
        r_losses = self.r_losses_
        null_values = self.null_values_
            
        if double_split:
            sizes = np.array(
                [len(l_loss) + len(r_loss)
                 for l_loss, r_loss in zip(l_losses, r_losses)])
        else:
            sizes = np.array(
                [len(l_loss) for l_loss in l_losses])
        
        estimates = np.array(
            [r_loss.mean() - l_loss.mean()
             for l_loss, r_loss in zip(l_losses, r_losses)])
        
        if (null_dist == "permutation") or (null_dist == "resampling"):
            std_errors = np.array(
                [null_value.std() for null_value in null_values])
            p_values = np.array(
                [(null_value > estimate).mean() 
                 for estimate, null_value in zip(estimates, null_values)])
        else:
            if double_split:
                std_errors = np.array(
                    [np.sqrt(
                        r_loss.var() / len(r_loss) + l_loss.var() / len(l_loss) ) 
                     for l_loss, r_loss in zip(l_losses, r_losses)])
            else:
                std_errors = np.array(
                    [np.sqrt((r_loss - l_loss).var() / len(l_loss)) 
                     for l_loss, r_loss in zip(l_losses, r_losses)])
            p_values = np.array(
                [1 - scipy.stats.norm.cdf(estimate / std_error) 
                 for estimate, std_error in zip(estimates, std_errors)])
        return sizes, estimates, std_errors, p_values


    def _combined_p_values(
        self,
        p_values
    ):
        const = np.sum(1 / (np.arange(len(p_values)) + 1))
        order_const = const * (len(p_values) / (np.arange(len(p_values)) + 1))
        t0 = np.mean(np.tan((.5 - np.array(p_values)) * np.pi))
        
        combined_p_values = {
            "gmean": np.e * scipy.stats.gmean(p_values, 0),
            "median": 2 * np.median(p_values, 0),
            "q1": len(p_values) / 2 * np.partition(p_values, 1)[1],
            "min": len(p_values) * np.min(p_values, 0),
            "hmean": np.e * np.log(len(p_values)) * scipy.stats.hmean(p_values, 0),
            "hommel": np.min(np.sort(p_values) * order_const),
            "cauchy": .5 - np.arctan(t0) / np.pi}
        combined_p_values = {key: np.minimum(value, 1) 
                             for key, value in combined_p_values.items()}
        return combined_p_values


    def _check(
        self
    ):
        learner = self.learner
        sampler = self.sampler
        competitor = self.competitor
        removal = self.removal
        method = self.method
        loss_func = self.loss_func
        null_dist = self.null_dist
        double_split = self.double_split
        perturb_size = self.perturb_size
        n_copies = self.n_copies
        n_permutations = self.n_permutations
        random_state = self.random_state

        if learner is None:
            raise ValueError(
                "Argument `learner` must be given.")
        if removal is None:
            raise ValueError(
                "Argument `removal` must be given.")
        else:
            if not isinstance(removal, (int, str, list)):
                raise ValueError(
                    "Support `removal` types are {}.".format(
                    {"int", "str", "list"}))
        if method is not None:
            if not isinstance(method, str):
                raise ValueError(
                    "Support `method` type is 'str'.")
        if loss_func is not None:
            if not loss_func in ["mean_squared_error", 
                                 "mean_absolute_error", 
                                 "zero_one_loss", 
                                 "log_loss"]:
                raise ValueError(
                    "Support `loss_func` values are {}".format(
                    {"mean_squared_error", 
                     "mean_absolute_error", 
                     "zero_one_loss", 
                     "log_loss"}))
        if null_dist is not None:
            if not null_dist in ["resampling", 
                                 "permutation", 
                                 "normality"]:
                raise ValueError(
                    "Support `null_dist` values are {}".format(
                    {"resampling", "permutation", "normality"}))
        if double_split is not None:
            if not isinstance(double_split, bool):
                raise ValueError(
                    "Support `loss_func` type is 'bool'.")
        if perturb_size is not None:
            if not isinstance(perturb_size, (int, float)):
                raise ValueError(
                    "Support `perturb_size` types are {}.".format(
                    {"int", "float"}))
        if n_copies is not None:
            if not isinstance(n_copies, int):
                raise ValueError(
                    "Support `n_copies` type is 'int'.")
        if n_permutations is not None:
            if not isinstance(n_permutations, int):
                raise ValueError(
                    "Support `n_permutations` type is 'int'.")
        if random_state is not None:
            if not isinstance(random_state, int):
                raise ValueError(
                    "Support `random_state` type is 'int'.")

    def _setup(
        self
    ):
        learner = self.learner
        method = self.method
        loss_func = self.loss_func
        null_dist = self.null_dist
        double_split = self.double_split
        perturb_size = self.perturb_size
        n_copies = self.n_copies
        n_permutations = self.n_permutations
        random_state = self.random_state

        if loss_func is None:
            if is_classifier(learner):
                loss_func = "log_loss"
            else:
                loss_func = "mean_squared_error"

        if loss_func == "log_loss":
            def log_loss(target, pred):
                eps = np.finfo(pred.dtype).eps
                pred = np.clip(pred, eps, 1 - eps)
                loss = -xlogy(target, pred).sum(axis=1)
                return loss
            loss_func = log_loss
            binarize = True
            response_method = "predict_proba" 

        if loss_func == "zero_one_loss":
            def zero_one_loss(target, pred):
                loss = 1 * (target == pred)
                return loss
            loss_func = zero_one_loss
            binarize = False
            response_method = "predict" 
        
        if loss_func == "mean_squared_error":
            def mean_squared_error(target, pred):
                loss = (target - pred)**2
                return loss
            loss_func = mean_squared_error
            binarize = False
            response_method = "predict"
        
        if loss_func == "mean_absolute_error":
            def mean_absolute_error(target, pred):
                loss = np.abs(target - pred)
                return loss
            loss_func = mean_absolute_error
            binarize = False
            response_method = "predict"

        is_inferred = False
        if isinstance(learner, Crosser):
            is_crosser = True
            n_splits = learner._n_splits
            n_repeats = learner._n_repeats
            n_folds = learner._n_folds
        else:
            is_crosser = False
            n_splits = 1
            n_repeats = 1
            n_folds = 1
        
        self.loss_func = loss_func
        self._binarize = binarize
        self._response_method = response_method
        self._is_inferred = is_inferred
        self._is_crosser = is_crosser
        self._n_splits = n_splits
        self._n_repeats = n_repeats
        self._n_folds = n_folds

class CIT(Inferer):
    def __init__(
        self, 
        learner,
        sampler,
        removal,
        method,
        *,
        loss_func = None,
        null_dist = None,
        n_copies = None,
        n_permutations = None,
        random_state = None
    ):
        super().__init__(
            learner = learner,
            sampler = sampler,
            removal = removal,
            method = method,
            loss_func = loss_func,
            null_dist = null_dist,
            n_copies = n_copies,
            n_permutations = n_permutations,
            random_state = random_state)
        
        self._check()
        self._setup()
        
    def infer(
        self,
        X,
        y
    ):
        learner = self.learner
        sampler = self.sampler
        removal = self.removal
        loss_func = self.loss_func
        null_dist = self.null_dist
        n_copies = self.n_copies
        n_permutations = self.n_permutations
        random_state = self.random_state
        is_crosser = self._is_crosser
        binarize = self._binarize
        response_method = self._response_method

        X_copy, y_copy = X.copy(), y.copy() 
        
        if is_crosser:
            l_features = learner._features(X_copy, y_copy)
            if isinstance(y, pd.Series):
                l_targets = learner._targets(X_copy, y_copy.values, binarize)
            else:
                l_targets = learner._targets(X_copy, y_copy, binarize)
            l_preds = [getattr(learner, response_method)(l_feature, split = split) 
                       for split, l_feature in enumerate(l_features)]
            l_losses = [loss_func(l_target, l_pred)
                       for l_target, l_pred in zip(l_targets, l_preds)]
        else:
            l_features = [X_copy]
            if is_classifier(learner) and binarize:
                label_binarizer = LabelBinarizer()
                if hasattr(learner, "classes_"):
                    _ = label_binarizer.fit(learner.classes_)
                else:
                    _ = label_binarizer.fit(sorted(set(y_copy)))
                l_targets = [label_binarizer.transform(y_copy)]
            else:
                if isinstance(y_copy, pd.Series):
                    l_targets = [y_copy.values]
                else:
                    l_targets = [y_copy]
            l_preds = [getattr(learner, response_method)(l_feature) 
                       for l_feature in l_features]
            l_losses = [loss_func(l_target, l_pred)
                        for l_target, l_pred in zip(l_targets, l_preds)]
        
        if isinstance(X_copy, pd.DataFrame):
            X_removed = X_copy.drop(removal, axis = 1)
            y_removed = X_copy[removal]
        else:
            X_removed = np.delete(X_copy, removal, axis = 1)    
            y_removed = X_copy[:, removal]
        
        r_features = l_features
        
        if isinstance(sampler, Crosser):
            rvs = sampler._rvs(
                X_removed,
                y_removed,
                n_copies = n_copies,
                random_state = random_state)
        else:
            if is_crosser:
                s_features = learner._features(
                    X_removed, 
                    y_removed)
                s_targets = learner._targets(
                    X_removed, 
                    y_removed)
            else:
                s_features = [X_removed]
                s_targets = [y_removed]
                
            rvs = [] 
            for s_feature, s_target in zip(s_features, s_targets):
                if hasattr(sampler, "sample"):
                    rv = sampler.sample(
                        s_feature, 
                        s_target,
                        n_copies = n_copies,
                        random_state = random_state)
                else:
                    rv = _sample(
                        sampler, 
                        s_feature, 
                        s_target,
                        n_copies = n_copies,
                        random_state = random_state)
                rvs.append(rv)
                
        r_losses = []

        def _r_loss_i(rv_i):
            if isinstance(r_feature, pd.DataFrame):
                r_feature.loc[:, removal] = rv_i
            else:
                r_feature[:, removal] = rv_i
            if isinstance(learner, Crosser):
                r_pred_i = learner.predict(
                    r_feature, 
                    split = split)
            else:
                r_pred_i = learner.predict(
                    r_feature)
            r_loss_i = loss_func(l_target, r_pred_i)
            return r_loss_i
        
        for split, (l_loss, l_target, r_feature, rv) in enumerate(
            zip(l_losses, l_targets, r_features, rvs)):
            r_loss = np.apply_along_axis(
                _r_loss_i,
                axis = 1,
                arr = rv)
            r_losses.append(r_loss)
        
        if null_dist == "resampling":
            null_values = []  
            for split, (l_loss, r_loss) in enumerate(
                zip(l_losses, r_losses)):
                null_value = r_loss.mean() - r_loss.mean(axis = 1)
                null_values.append(null_value)
                r_loss = r_loss.mean(axis = 0)
                r_losses[split] = r_loss
        else:
            for split, (l_loss, r_loss) in enumerate(
                zip(l_losses, r_losses)):
                r_loss = r_loss.mean(axis = 0)
                r_losses[split] = r_loss
            
            if null_dist == "permutation":
                null_values = [] 
                rng = np.random.default_rng(random_state)
                for l_loss, r_loss in zip(l_losses, r_losses):
                    paired_loss = np.column_stack([l_loss, r_loss])
                    null_value = []
                    for permutation in range(n_permutations):
                        permuted_loss = rng.permuted(
                            paired_loss, 
                            axis = 1)
                        null_value.append(
                            permuted_loss[:,0].mean() - 
                            permuted_loss[:,1].mean())
                    null_value = np.array(null_value)
                    null_values.append(null_value)
            else:
                null_values = None

        self.l_losses_ = l_losses
        self.r_losses_ = r_losses
        self.null_values_ = null_values
        self._is_inferred = True

    def _check(
        self
    ):
        super()._check()
        learner = self.learner
        sampler = self.sampler
        removal = self.removal
        method = self.method
        
        if sampler is None:
            raise ValueError(
                "When implementing `CIT`, "
                "`sampler` must be given.")
        if isinstance(learner, Crosser):
            if isinstance(sampler, Crosser):
                if not learner.cv == sampler.cv:
                    raise ValueError(
                        "When `learner` and `sampler` are both `Crosser`, "
                        "they must share the same cross-validator.")  
        else:
            if isinstance(sampler, Crosser):
                raise ValueError(
                    "When `learner` is not `Crosser`, "
                    "`sampler` must be usual `Estimator`.")  
        if isinstance(removal, list):
            raise ValueError(
                "When implementing `CIT`, "
                "`removal` must be `int` or `str`.")
        if not method in ["HRT", "RPT", "CPI"]:
            raise ValueError(
                    "Support `method` values are {}".format(
                    {"HRT", "RPT", "CPI"}))
  
    def _setup(
        self
    ):
        super()._setup()
        method = self.method
        null_dist = self.null_dist
        n_copies = self.n_copies
        n_permutations = self.n_permutations
        random_state = self.random_state

        if method is None:
            method = "RPT"
        if method in ["HRT","RPT"]:
            if null_dist is None:
                null_dist = "resampling"
        elif method in ["CPI"]:
            if null_dist is None:
                null_dist = "normality"
        if null_dist in ["resampling"]:
             if n_copies is None:
                n_copies = 2000       
        elif null_dist in ["normality", "permutation"]:
            if n_copies is None:
                n_copies = 1            
            if null_dist in ["permutation"]:
                if n_permutations is None:
                    n_permutations = 2000
        if null_dist in ["normality", "resampling"]:
            if n_permutations is not None:
                warnings.warn(
                    "When `null_dist` is 'normality' or 'resampling', "
                    "`n_permutations` must be `None`.")
                n_permutations = None
        
        self.method = method 
        self.null_dist = null_dist
        self.n_copies = n_copies
        self.n_permutations = n_permutations
        self.random_state = random_state

class RIT(Inferer):
    def __init__(
        self, 
        learner,
        competitor,
        removal,
        method,
        *,
        loss_func = None,
        null_dist = None,
        double_split = None,
        perturb_size = None,
        n_permutations = None,
        random_state = None):
        super().__init__(
            learner = learner,
            competitor = competitor,
            removal = removal,
            method = method,
            loss_func = loss_func,
            null_dist = null_dist,
            double_split = double_split,
            perturb_size = perturb_size,
            n_permutations = n_permutations,
            random_state = random_state)
        
        self._check()
        self._setup()
    
    def infer(
        self,
        X,
        y
    ):
        learner = self.learner
        competitor = self.competitor
        removal = self.removal
        loss_func = self.loss_func
        null_dist = self.null_dist
        n_permutations = self.n_permutations
        double_split = self.double_split
        perturb_size = self.perturb_size
        random_state = self.random_state
        is_crosser = self._is_crosser
        binarize = self._binarize
        response_method = self._response_method
        
        X_copy, y_copy = X.copy(), y.copy() 
        if is_crosser:
            l_features = learner._features(X_copy, y_copy)
            if isinstance(y_copy, pd.Series):
                l_targets = learner._targets(X_copy, y_copy.values, binarize)
            else:
                l_targets = learner._targets(X_copy, y_copy, binarize)
            l_preds = [getattr(learner, response_method)(
                l_feature, split = split) 
                       for split, l_feature in enumerate(l_features)]
            l_losses = [loss_func(l_target, l_pred)
                       for l_target, l_pred in zip(l_targets, l_preds)]
        else:
            l_features = [X_copy]
            if is_classifier(learner) and binarize:
                label_binarizer = LabelBinarizer()
                if hasattr(learner, "classes_"):
                    _ = label_binarizer.fit(learner.classes_)
                else:
                    _ = label_binarizer.fit(sorted(set(y_copy)))
                target = label_binarizer.transform(y_copy)
                if target.shape[1] == 1:
                    target = np.append(1 - target, target, axis=1)    
                l_targets = [target]
            else:
                if isinstance(y_copy, pd.Series):
                    l_targets = [y_copy.values]
                else:
                    l_targets = [y_copy]
            l_preds = [getattr(learner, response_method)(l_feature) 
                       for l_feature in l_features]
            l_losses = [loss_func(l_target, l_pred)
                        for l_target, l_pred in zip(l_targets, l_preds)]

        if isinstance(X_copy, pd.DataFrame):
            X_removed = X_copy.drop(removal, axis = 1)
        else:
            X_removed = np.delete(X_copy, removal, axis = 1)   

        if isinstance(competitor, Crosser):
            r_features = competitor._features(X_removed, y_copy)
            r_targets = competitor._targets(X_removed, y_copy, binarize)
            r_preds = [getattr(competitor, response_method)(
                r_feature, split = split) 
                       for split, r_feature in enumerate(r_features)]
            r_losses = [loss_func(r_target, r_pred)
                       for r_target, r_pred in zip(r_targets, r_preds)]
        else:
            if is_crosser:
                r_features = learner._features(X_removed, y_copy)
                r_targets = learner._targets(X_removed, y_copy, binarize)
            else:        
                r_features = [X_removed]
                if is_classifier(competitor):
                    if binarize:
                        target = label_binarizer.transform(y_copy)
                        if target.shape[1] == 1:
                            target = np.append(1 - target, target, axis=1)    
                        r_targets = [target]
                    else:
                        r_targets = [y_copy]
                else:
                    r_targets = [y_copy]
            r_preds = [getattr(competitor, response_method)(r_feature) 
                       for r_feature in r_features]
            r_losses = [loss_func(r_target, r_pred)
                       for r_target, r_pred in zip(r_targets, r_preds)]        

        if double_split:
            l_losses = [l_loss[[i for i in range(len(l_loss)) if i % 2 == 0]] 
                        for l_loss in l_losses]
            r_losses = [r_loss[[i for i in range(len(r_loss)) if i % 2 == 1]] 
                        for r_loss in r_losses]

        if perturb_size is not None:
            rng = np.random.default_rng(random_state)
            r_losses = [r_loss + rng.normal(
                scale = perturb_size, 
                size = len(r_loss)) for r_loss in r_losses]
            
        if null_dist == "permutation":
            null_values = [] 
            rng = np.random.default_rng(random_state)
            if double_split:
                for l_loss, r_loss in zip(l_losses, r_losses):
                    concated_loss = np.concatenate([l_loss, r_loss])
                    size = len(concated_loss)
                    null_value = []
                    for permutation in range(n_permutations):
                        permuted_loss = rng.permuted(concated_loss)
                        null_value.append(
                            permuted_loss[:math.ceil(size/2)].mean() - 
                            permuted_loss[math.ceil(size/2):].mean())
                    null_value = np.array(null_value)
                    null_values.append(null_value)                
            else:
                for l_loss, r_loss in zip(l_losses, r_losses):
                    paired_loss = np.column_stack([l_loss, r_loss])
                    null_value = []
                    for permutation in range(n_permutations):
                        permuted_loss = rng.permuted(
                            paired_loss, 
                            axis = 1)
                        null_value.append(
                            permuted_loss[:,0].mean() - 
                            permuted_loss[:,1].mean())
                    null_value = np.array(null_value)
                    null_values.append(null_value)
        else:
            null_values = None

        self.l_losses_ = l_losses
        self.r_losses_ = r_losses
        self.null_values_ = null_values
        self._is_inferred = True


    def _check(
        self
    ):
        super()._check()
        learner = self.learner
        competitor = self.competitor
        removal = self.removal
        method = self.method
        
        if competitor is None:
            raise ValueError(
                "When implementing `RIT`, "
                "`competitor` must be given.")
        if isinstance(learner, Crosser):
            if isinstance(competitor, Crosser):
                if not learner.cv == competitor.cv:
                    raise ValueError(
                        "When `learner` and `competitor` are both `Crosser`, "
                        "they must share the same cross-validator.")
        else:
            if isinstance(competitor, Crosser):
                raise ValueError(
                    "When `learner` is not `Crosser`, "
                    "`competitor` must be usual `Estimator`.")                  
        if isinstance(removal, list):
            for removal_i in removal:
                if not isinstance(removal_i, (int, str)):
                    raise ValueError(
                        "When `removal` is `list`, "
                        "its element must be `int` or `str`.")
        if not method in ["LOCO", "PIE"]:
            raise ValueError(
                    "Support `method` values are {}".format(
                    {"LOCO", "PIE"}))
  
    def _setup(
        self
    ):
        super()._setup()
        method = self.method
        null_dist = self.null_dist
        double_split = self.double_split
        perturb_size = self.perturb_size
        n_permutations = self.n_permutations
        random_state = self.random_state

        if method is None:
            method = "PIE"
        if null_dist is None:
            null_dist = "normality"
        else:
            if null_dist in ["permutation"]:
                if n_permutations is None:
                    n_permutations = 2000
            elif null_dist in ["resampling"]:
                warnings.warn(
                    "When `method` is 'LOCO' or 'PIE', "
                    "`null_dist` must be 'normality' or 'permutation'.")
        if double_split is None:
            if method in ["PIE"]:
                double_split = True
            else:
                double_split = False
        if null_dist in ["normality"]:
            if n_permutations is not None:
                warnings.warn(
                        "When `null_dist` is 'normality', "
                        "`n_permutations` must be `None`.")
                n_permutations = None

        self.method = method 
        self.null_dist = null_dist
        self.double_split = double_split
        self.perturb_size = perturb_size
        self.n_permutations = n_permutations
        self.random_state = random_state

