# Python3 program to implement traveling salesman
# problem using naive approach.
from sys import maxsize
from itertools import permutations
import numpy as np
#from scipy.spatial import distance
from math import dist

# implementation of traveling Salesman Problem
def travellingSalesmanProblem(trajectory, s=0):
    graph = path_cost(trajectory = trajectory)
    V = len(graph)
    # store all vertex apart from source vertex
    vertex = []
    for i in range(V):
        if i != s:
            vertex.append(i)
 
    # store minimum weight Hamiltonian Cycle
    min_path = maxsize
    next_permutation=permutations(vertex)
    for i in next_permutation:
 
        # store current Path weight(cost)
        current_pathweight = 0
        current_path = []
 
        # compute current path weight
        k = s
        current_path.append(k)
        for j in i:
            current_pathweight += graph[k][j]
            k = j
            current_path.append(k)
        current_pathweight += graph[k][s]
 
        # update minimum and path
        if current_pathweight < min_path:
            path = current_path
            min_path = current_pathweight

    optimal_path = []
    for i in path:
        optimal_path.append(trajectory[i])

    #optimal_path.append([10.6,90,7])

    print('optimal_path = {}'.format(optimal_path))

    return optimal_path
 
def path_cost(trajectory):
    # Creates a graph of the path cost for each combination of points
    graph = []

    for point in trajectory:
        cost = []
        for destiny in trajectory:
            #dst = distance.euclidean(point, destiny)
            dst = dist(point, destiny)
            cost.append(dst)
        graph.append(cost)
    return graph

# Driver Code
if __name__ == "__main__":

    # Change this list
    trajectory = [[2,3,1],[3,4,1],[14,4,1],[3,34,1],[3,4,21],[3.7,3.5,4],[3,4,3]]

    optimal_path = travellingSalesmanProblem(trajectory= trajectory)

