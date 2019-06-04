import os
import re
import json
import yaml
import csv
import time
from glob import glob
from copy import copy, deepcopy

import numpy as np
from numpy import array, ndarray, matrix, arange, ogrid, meshgrid, reshape, tile, repeat, squeeze, roll, stack, concatenate as cat, \
flip, linspace, zeros, ones, eye, outer, inner, dot, matmul, trace, transpose, swapaxes, expand_dims, ndim, isnan,\
allclose, isscalar, where
from numpy import empty_like, zeros_like, ones_like
from numpy import mean, std, var, sort, argsort, histogram, unravel_index
from numpy import exp, log, log2, sin, cos, tan,\
    arcsin as asin, arccos as acos, arctan2 as atan2,\
    sqrt, floor, ceil, round, conj, abs, sum, prod, real, imag, sign,\
    rad2deg, deg2rad
from numpy import int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64, complex128
from numpy import pi
from numpy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift
from numpy.linalg import inv, pinv, solve, eig, svd, norm
from numpy.random import rand, randn, randint 
np.set_printoptions(
        precision = 3,
        floatmode = 'fixed',
        suppress = True)
#np.seterr(all='raise')

def min(x, y = None):
    if y is not None:
        return np.minimum(x,y)
    return np.min(x)

def max(x, y = None):
    if y is not None:
        return np.maximum(x,y)
    return np.max(x)

def normalize(array, norm = norm):
    """
    Normalizes an array according to a the specified norm.
    
    Parameters
    ----------
    array: array 
        Array to normalize
    norm : what
        Norm to normalize against.
        Default is the euclidean norm
    Returns
    -------
    array
        Normalized array
    """
    return array / norm(array)

def shuffle(array, axis = 0):
    """
    Shuffles an array along the specified axis
    
    Parameters
    ----------
    array : array 
        Array to shuffle 
    axis : what
        Axis to shuffle over
        Default is the first dimension of the array 
    Returns
    -------
    array
        Shuffled array
    """
    tmp = array.copy()
    np.random.shuffle(tmp)
    return tmp

def argmin(X):
    """
    Computes the index of the minimum element of an array
    
    Parameters
    ----------
    X : array
        Array to calculate the minimum over
    Returns
    -------
    tuple or int
        (Multi-)index of the minimum element
    """
    amin = unravel_index(np.argmin(X), np.array(X).shape)
    if len(amin) == 1:
        amin = amin[0]
    return amin
def argmax(X):
    amax = unravel_index(np.argmax(X), np.array(X).shape)
    if len(amax) == 1:
        amax = amax[0]
    return amax

import scipy
#from scipy.ndimage import convolve as conv
from scipy.signal import convolve as conv, convolve2d as conv2
from scipy.stats import entropy

def cconv(x,w):
    return scipy.ndimage.convolve(x,w,mode='wrap')

def cconv1(x,w,axis=0):
    return scipy.ndimage.convolve1d(x,w,axis,mode='wrap')

import cv2 as cv
from cv2 import imread, imwrite,\
    waitKey as wait_key, destroyWindow as destroy_window, destroyAllWindows as destroy_all_windows
from cv2 import pyrUp as pyr_up, pyrDown as pyr_down

def imshow(window_name, image):
    dtype = image.dtype
    if dtype in [float32, float64]:
        normalized = (image - image.min()) / (image.max() - image.min()) 
        cv.imshow(window_name, normalized)
    elif dtype == uint8:
        cv.imshow(window_name, image)
    else:
        raise Exception("Don't know how to imshow image with dtype: {}".format(dtype))

def imsave(image, path):
    dtype = image.dtype
    if dtype in [float32, float64]:
        image = (image - image.min()) / (image.max() - image.min()) 
        image = (255*image).astype(uint8)
    elif dtype == uint8:
        pass
    else:
        raise Exception("Don't know how to imshow image with dtype: {}".format(dtype))
    cv.imwrite(path, image)

def resize(x,shape):
    return cv.resize(x,(shape[1],shape[0]), 0, 0, cv.INTER_LINEAR)

def imshowgray(window_name, gray_image):
    gray_image = gray_image - gray_image.min()
    gray_image = gray_image / gray_image.max()
    imshow(window_name, gray_image)

def _ensure_array(possibly_cvmat):
    if not isinstance(possibly_cvmat, ndarray):
        return possibly_cvmat.get()
    return possibly_cvmat

def draw_circle(image, *args):
    out = cv.circle(image, *args)
    return _ensure_array(out)

def draw_line(image, *args):
    out = cv.line(image, *args)
    return _ensure_array(out)

def draw_rectangle(image, *args):
    out = cv.rectangle(image, *args)
    return _ensure_array(out)

import matplotlib as mpl
# uncomment if want to use tkinter
#mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, subplots, show, draw, pause, plot, bar, scatter, legend,\
        suptitle as title, xlabel, ylabel, axis, xlim, ylim, xticks, yticks, savefig, clf
#from mpl_toolkits.mplot3d import Axes3D

def rgb2gray(x):
    return (sum(x,axis=2) / 3).astype(x.dtype)

def readfile(filepath):
    with open(filepath, 'r') as f:
        return f.read()
    
def csvread(filename, parse_float = True):
    sniffer = csv.Sniffer()
    with open(filename, 'r') as f:
        content = f.read()
    dialect = sniffer.sniff(content, delimiters = ',;\t ')
    with open(filename, 'r') as f:
        reader = csv.reader(f, dialect)
        if parse_float:
            return [[float(x) for x in line] for line in reader]
        return [[x for x in line] for line in reader]

def mkdirp(path):
    if os.path.isfile(path):
        path = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path)

def tic():
    tic.time = time.clock()

def toc():
    toc.time = time.clock()
    print(toc.time - tic.time)
