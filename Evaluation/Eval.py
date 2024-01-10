#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Referred to https://github.com/scipy/scipy/blob/c1ed5ece8ffbf05356a22a8106affcd11bd3aee0/scipy/linalg/_sketches.py#L14
import numpy as np
import torch
from scipy._lib._util import check_random_state, rng_integers
from scipy.sparse import csc_matrix

#Read video in
import cv2
from matplotlib import pyplot as plt


# In[48]:


def cwt_matrix(n_rows, n_columns, n, seed=None):
    r"""
    Generate a matrix S which represents a Clarkson-Woodruff transform.
    Given the desired size of matrix, the method returns a matrix S of size
    (n_rows, n_columns) where each column has all the entries set to 0
    except for one position which has been randomly set to +1 or -1 with
    equal probability.
    Parameters
    ----------
    n_rows : int
        Number of rows of S
    n_columns : int
        Number of columns of S
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
    Returns
    -------
    S : (n_rows, n_columns) csc_matrix
        The returned matrix has ``n_columns`` nonzero entries.
    Notes
    -----
    Given a matrix A, with probability at least 9/10,
    .. math:: \|SA\| = (1 \pm \epsilon)\|A\|
    Where the error epsilon is related to the size of S.
    """
    rng = check_random_state(seed)
    rows = rng_integers(rng, 0, n_rows, n_columns)
    #Every column has exactly one non-zero element
    cols = np.arange(n_columns)
    signs = torch.Tensor(rng.choice([1, -1], nnz))
    #signs=data, and data[cols[i]:cols[i+1]]=non-zero element value 
    #rows[cols[i]:cols[i+1]]=row index of non-zero elements
    #cols[i+1]-col[i]=number of non-zero elements in col i
    S = csc_matrix((signs, rows, cols),shape=(n_rows, n_columns))
    return S


def initial_matrix(n_rows, n_columns, n, seed=None):
    rng = check_random_state(seed)
    # number of non-zero entries >= 1 per column
    nnz = n_columns*n
    # random indices to place non zero entries
    rows = rng_integers(rng, 0, n_rows, nnz)
    cols = rng_integers(rng, 0, n_columns, nnz)
    # values in \pm 1
    signs = torch.Tensor(rng.choice([1, -1], nnz))
    indices = torch.stack((torch.Tensor(rows), torch.Tensor(cols)))
    S = torch.sparse_coo_tensor(indices=indices, values=signs, size=(n_rows, n_columns))
    return S


def approx_A(A,m=20,k=10,method="CW",seed=None):
    r"""
    Return an approximate matrix of A based on the given parameters.
    Parameters
    ----------
    A : array
        The matrix to be approximated.
    m : int
        The number of rows of the sketching matrix S, which will be calculated in this function
    k : int
        The best-k approximation of A
    method : str
        The method used to generate the sketching matrix S, which can be "CW", "Gaussian"
        or "Train"
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
    Returns
    -------
    S : (n_rows, n_columns) csc_matrix
        The returned matrix has ``n_columns`` nonzero entries.
    A_approx: array
        The best-k approximation of A, shape = A.shape
    """
    #Generate sketching matrix S
    if method=="CW":
        S = cwt_matrix(m, A.shape[0], seed=rng).toarray()
        _,_,vt = np.linalg.svd(S@A, full_matrices=False)
        r = np.linalg.matrix_rank(S@A)
        vt = vt[:r,:]
        AV = A @ vt.T
        uk,sk,vtk = np.linalg.svd(AV, full_matrices=False)
        uk = uk[:,:k]
        sk = sk[:k]
        sk = np.diag(sk)
        vtk = vtk[:k,:]
        AV_approx = uk @ sk @ vtk
        A_approx = AV_approx @ vt
        return S, A_approx
    return None

def eval_metric(A,A_approx,k=10):
    r"""
    Return an metric used in the paper.
    Parameters
    ----------
    A : array (height, width,number of frames)
        The matrix to be approximated.
    A_approx : array (height, width, number of frames)
        The best-k approximation of A, shape = A.shape
    k : int
        The best-k approximation of A
    Returns
    -------
    Error : float
        The error between A and A_approx. Formula follows section 5 in the paper.
    """
    #Best k approximation of A by only using SVD
    A_approx_svd = np.zeros(A.shape)
    for i in range(A.shape[2]):
        u,s,vt = np.linalg.svd(A[:,:,i], full_matrices=False)
        u = u[:,:k]
        s = s[:k]
        s = np.diag(s)
        vt = vt[:k,:]
        A_approx_svd.append(u @ s @ vt)
    A_approx_svd = np.stack(A_approx_svd)
    #Error
    Appte = A - A_approx_svd
    Appt = np.mean(np.linalg.norm(Appte, axis=(0,1)))
    error = A - A_approx
    error = np.mean(np.linalg.norm(error, axis=(0,1))) - Appt
    return error


# In[56]:

# cap = cv2.VideoCapture('Dataset/MIT.mp4')
# ref, A = cap.read()
# A = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
# # might need this step for normalization
# A = A.astype(np.float64) / 255.0
# cap.release()
# # list parameter of read-in video
# print(A.shape)


# # In[62]:

# rng = np.random.default_rng()
# S, A_approx = approx_A(A)
# print(eval_metric(A, A_approx))
