import numpy as np
import math

def find_index(x, X):
    ''' works for continous arrays that either only increase or decrease'''
    dif = abs(np.array(X)-x).tolist()
    return dif.index(min(dif))

def single_sigfig(x, sf):
    if x in [np.inf, np.nan, 0]:
        return x
    else:
        return np.round(x, sf-math.floor(math.log10(abs(x)))-1)
              
def sigfig(X, sf):
    try:
        Xs = []
        for x in X:
            Xs.append(single_sigfig(x, sf))
        return np.asarray(Xs)
    except:
        return single_sigfig(X, sf)
    
def rel_text(ax, x, y, text, fontsize=14):
    xbot, xtop = ax.get_xlim()
    ybot, ytop = ax.get_ylim()
    xpos = (xtop-xbot)*x + xbot
    ypos = (ytop-ybot)*y + ybot
    ax.text(xpos, ypos, text, va='center', ha='center', fontsize=fontsize)

def dead_pixels(data, m):
    '''Feed data into here to change the pixels 
    that have saturated to an average value
    Can be used for 2D or 3D data'''
    if data.ndim == 3:
        for i in range(data.shape[0]):
            data_slice = data[i]
            a, b = data_slice.shape
            flat = data_slice.flatten()
            indexing = np.argsort(flat)
            sort = np.sort(flat)
            Max = sort[-m-1]
            for j in range(m):
                flat[indexing[-j-1]] = Max
            data_slice = flat.reshape((a,b))
            data[i] = data_slice
    elif data.ndim == 2:
        a, b = data.shape
        flat = data.flatten()
        indexing = np.argsort(flat)
        sort = np.sort(flat)
        Max = sort[-m-1]
        for j in range(m):
            flat[indexing[-j-1]] = Max
        data = flat.reshape((a,b))
    else:
        print('I havent got this far')
    return data
