import numpy as np
import argparse
import logging
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import itertools
from scipy.interpolate import LinearNDInterpolator

    # phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_partial_repr(input_points, control_points):
    N = len(input_points)
    M = len(control_points)
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
    # original implementation, very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    return repr_matrix

def tps(boundary):
    M = 128
    boundary = boundary.permute(0,2,3,1)
    boundary = boundary[0].cpu().detach().numpy()
    t = boundary[0,:-1,:]
    r = boundary[:-1,-1,:]
    d = boundary[-1,1:,:][::-1]
    l = boundary[1:,0,:][::-1]
    source_control_points = np.concatenate((t[:,::-1], r[:,::-1], d[:,::-1], l[:,::-1]), axis=0).reshape((-1,2))
    
    tt, rr, dd, ll= np.arange(-1, 1 + 0.00001, 2.0 / (M - 1)),np.arange(-1, 1 + 0.00001, 2.0/(M - 1)),-np.arange(-1, 1 + 0.00001, 2.0 / (M - 1)),-np.arange(-1, 1 + 0.00001, 2.0 / (M- 1))
    ttt, rrr, ddd, lll = np.array([[-1,tt1] for tt1 in tt]),np.array([[rr1,1] for rr1 in rr]),np.array([[1,dd1] for dd1 in dd]),np.array([[ll1,-1] for ll1 in ll])

    target_control_points = np.concatenate((ttt[:-1],rrr[:-1], ddd[:-1], lll[:-1]), axis=0).reshape((-1,2))
    zz = 4
    target_control_points = np.array([target_control_points[i] for i in range(0,len(target_control_points),zz)])
    source_control_points = np.array([source_control_points[i] for i in range(0,len(source_control_points),zz)])


    target_control_points,source_control_points  = torch.Tensor(target_control_points),torch.Tensor(source_control_points)
    N = len(target_control_points)
    forward_kernel = torch.zeros(N + 3, N + 3)
    
    target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
    forward_kernel[:N, :N].copy_(target_control_partial_repr)
    forward_kernel[:N, -3].fill_(1)
    forward_kernel[-3, :N].fill_(1)
    forward_kernel[:N, -2:].copy_(target_control_points)
    forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))

    # compute inverse matrix
    inverse_kernel = torch.inverse(forward_kernel)

    # create target cordinate matrix
    Mesh = 128
    HW = Mesh * Mesh
    target_coordinate = list(itertools.product(range(Mesh), range(Mesh)))
    target_coordinate = torch.Tensor(target_coordinate)* 2/(Mesh - 1) - 1  # HW x 2

    target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)
    target_coordinate_repr = torch.cat([target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate], dim=1)

    Y = torch.cat([source_control_points, torch.zeros(3, 2)], dim =0)

    mapping_matrix = torch.matmul(inverse_kernel, Y)
    source_coordinate = torch.matmul(target_coordinate_repr, mapping_matrix)
    
    grid = source_coordinate.view(Mesh,Mesh, 2)[:,:,[1,0]]
    return grid