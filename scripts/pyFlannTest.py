#!/usr/bin/env python
import math
import pyflann
import numpy as np
from numpy import *
from numpy.random import *



if __name__ == '__main__':
    nn_solver = pyflann.FLANN()
    #model_arr = rand(10,1)
    #sparse_contour = rand(4,1)
    model_arr = np.array([[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[1,7]],dtype=float)
    sparse_contour = np.array([[2,1],[4,1],[6,1]],dtype=float)
    result,dists = nn_solver.nn(model_arr,sparse_contour, num_neighbors=1,algorithm="kmeans",branching=32, iterations=3, checks=16);
    
    print "model : " + str(model_arr)
    print "sparse_contour : " + str(sparse_contour)
    print "dist : " + str(dists)
    # dist should be 1,3,5 but res is 1,9,25.. It uses eucledian norm such that |AB|=(a_x-b_x)^2 + (a_y-b_y)^2
