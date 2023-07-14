# -*- coding: utf-8 -*-
"""
@author: Md Tauhidul Islam, Postdoc, Xing Lab,
Department of Radiation Oncology, Stanford University
 
"""


import numpy as np
from scipy import linalg
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import check_array
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.utils.validation import _deprecate_positional_args

def _class_means(X, y):
    """Compute class means.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.

    Returns
    -------
    means : array-like of shape (n_classes, n_features)
        Class means.
    """
    classes, y = np.unique(y, return_inverse=True)
    cnt = np.bincount(y)
    means = np.zeros(shape=(len(classes), X.shape[1]))
    np.add.at(means, y, X)
    means /= cnt[:, None]
    return means


class ClassDiscriminative_OPT(BaseEstimator, LinearClassifierMixin,
                                 TransformerMixin):
    @_deprecate_positional_args
    def __init__(self, *, solver='svd', shrinkage=None, priors=None,
                 n_components=None, store_covariance=False, tol=1e-4):

        self.n_components = n_components
        self.tol = tol  # used only in svd solver

    def _solve_svd(self, X, y):
        """SVD solver for the second optimization in CCIF. There are two SVD
        operations in this optimization

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        """
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        self.means_ = _class_means(X, y)
        
        # First SVD of second optimization in CCIF
        
        U, S, V = linalg.svd(X, full_matrices=False,lapack_driver='gesvd')
        V=V.T
        rank = np.sum(S > self.tol)
        U=U[:,:rank]
        V=V[:,:rank]       
        S=S[:rank]
        
        Sx=np.diag(1./S)
        eigVecStep1=np.matmul(V,Sx)
        
        _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
        binC = np.bincount(y_t)        
        Hb=_class_means(U, y)
        
        for ind in range(Hb.shape[0]):            
            Hb[ind,:]=Hb[ind,:]*np.sqrt(binC[ind])
            
        
        # Second SVD of second optimization in CCIF        
        Uh, Sh, eigVecStep2 = linalg.svd(Hb, full_matrices=True,lapack_driver='gesvd')        
        eigVecCCIF=np.matmul(eigVecStep1,eigVecStep2.T)
        
        for idx in range(eigVecCCIF.shape[1]):            
            eigVecCCIF[:,idx]=eigVecCCIF[:,idx]/np.linalg.norm(eigVecCCIF[:,idx])
            
        self.scalings_ =eigVecCCIF
        
        

 

    def fit(self, X, y):
        """Fit class-discriminative optimization model 

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data.

        y : array-like of shape (n_samples,)
            Target values.
        """
        X, y = self._validate_data(X, y, ensure_min_samples=2, estimator=self,
                                   dtype=[np.float64, np.float32])
        self.classes_ = unique_labels(y)
        n_samples, _ = X.shape
        n_classes = len(self.classes_)

        if n_samples == n_classes:
            raise ValueError("The number of samples must be more "
                             "than the number of classes.")

        # Maximum number of components no matter what n_components is
        # specified:
        max_components = X.shape[1]

        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than min(n_features, "
                    "n_classes - 1)."
                )
            self._max_components = self.n_components

        self._solve_svd(X, y)
        return self

    def transform(self, X):
        """Project data to maximize class separation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.
        """

        X = check_array(X)        
        X_new = np.dot(X, self.scalings_)
        return X_new[:, :self._max_components]
