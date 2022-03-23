from scipy.sparse.linalg import spsolve

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

import tqdm
from scipy.interpolate import LinearNDInterpolator
import torch
import itertools
import multiprocessing as mp
import scipy.sparse as sp

global grid
def grid(left,right,top,bottom,line,n):
    n1,n2 = left.shape[0],right.shape[0]
    n3,n4 = top.shape[0],bottom.shape[0]
    num = 0 
    for line_p in line:
        num+=len(line_p)
    N = n*n
    
    z = 0
    row = [];col = [];data = []

    for i in range(1,n-1):
        for j in range(1,n-1):
            row.append(z);col.append(i*n+j-1);data.append(1)
            row.append(z);col.append((i-1)*n+j);data.append(1)
            row.append(z);col.append(i*n+j+1);data.append(1)
            row.append(z);col.append((i+1)*n+j);data.append(1)
            row.append(z);col.append(i*n+j);data.append(-4)
            z+=1
            
    A = sp.coo_matrix((data, (row, col)), shape=(z,N))
    b = np.zeros((n1+n2,1))
    z =   0
    row = [];col = [];data = []

    for i in range(n1): 
        x,x1,y,y1 = int(left[i,0]),left[i,0]-int(left[i,0]),int(left[i,1]),left[i,1]-int(left[i,1])
        w1,w2,w3,w4 = (1-x1)*(1-y1),y1*(1-x1),x1*(1-y1),x1*y1
        
        row.append(z);col.append(x*n+y);data.append(w1)
        row.append(z);col.append(x*n+y+1);data.append(w2)
        row.append(z);col.append(x*n+y+n);data.append(w3)
        row.append(z);col.append(x*n+y+n+1);data.append(w4)
        b[z] = i/(n1-1)
        z = z+1

    for i in range(n2): 
        x,x1,y,y1 = int(right[i,0]),right[i,0]-int(right[i,0]),int(right[i,1]),right[i,1]-int(right[i,1])
        w1,w2,w3,w4 = (1-x1)*(1-y1),y1*(1-x1),x1*(1-y1),x1*y1
        row.append(z);col.append(x*n+y);data.append(w1)
        row.append(z);col.append(x*n+y+1);data.append(w2)
        row.append(z);col.append(x*n+y+n);data.append(w3)
        row.append(z);col.append(x*n+y+n+1);data.append(w4)
        b[z] = i/(n2-1)
        z = z+1
    B=sp.coo_matrix((data, (row, col)), shape=(n1+n2,N))

    c = np.zeros((n3+n4,1))
    z = 0
    row = [];col = [];data = []

    for i in range(n3): 
        x,x1,y,y1 = int(top[i,0]),top[i,0]-int(top[i,0]),int(top[i,1]),top[i,1]-int(top[i,1])
        w1,w2,w3,w4 = (1-x1)*(1-y1),y1*(1-x1),x1*(1-y1),x1*y1
        row.append(z);col.append(x*n+y);data.append(w1)
        row.append(z);col.append(x*n+y+1);data.append(w2)
        row.append(z);col.append(x*n+y+n);data.append(w3)
        row.append(z);col.append(x*n+y+n+1);data.append(w4)
        z = z+1

    for i in range(n4): 
        x,x1,y,y1 = int(bottom[i,0]),bottom[i,0]-int(bottom[i,0]),int(bottom[i,1]),bottom[i,1]-int(bottom[i,1])
        w1,w2,w3,w4 = (1-x1)*(1-y1),y1*(1-x1),x1*(1-y1),x1*y1
        row.append(z);col.append(x*n+y);data.append(w1)
        row.append(z);col.append(x*n+y+1);data.append(w2)
        row.append(z);col.append(x*n+y+n);data.append(w3)
        row.append(z);col.append(x*n+y+n+1);data.append(w4)
        c[z] = 1
        z = z+1

    C=sp.coo_matrix((data, (row, col)), shape=(n3+n4,N))
    
    if num:
        z = 0
        row = [];col = [];data = []
        for line_p in line:
            line_p = line_p[:,::-1]
            for i in range(len(line_p)-1):
                x,x1,y,y1 = int(line_p[i,0]),line_p[i,0]-int(line_p[i,0]),int(line_p[i,1]),line_p[i,1]-int(line_p[i,1])
                w1,w2,w3,w4 = (1-x1)*(1-y1),y1*(1-x1),x1*(1-y1),x1*y1

                x_,x1_,y_,y1_ = int(line_p[i+1,0]),line_p[i+1,0]-int(line_p[i+1,0]),int(line_p[i+1,1]),line_p[i+1,1]-int(line_p[i+1,1])
                w1_,w2_,w3_,w4_ = (1-x1_)*(1-y1_),y1_*(1-x1_),x1_*(1-y1_),x1_*y1_

                row.append(z);col.append(x*n+y);data.append(w1)
                row.append(z);col.append(x*n+y+1);data.append(w2)
                row.append(z);col.append(x*n+y+n);data.append(w3)
                row.append(z);col.append(x*n+y+n+1);data.append(w4)
                row.append(z);col.append(x_*n+y_);data.append(-w1_)
                row.append(z);col.append(x_*n+y_+1);data.append(-w2_)
                row.append(z);col.append(x_*n+y_+n);data.append(-w3_)
                row.append(z);col.append(x_*n+y_+n+1);data.append(-w4_)
                z+=1
        D=sp.coo_matrix((data, (row, col)), shape=(num,N))
        
    z = 0
    row = [];col = [];data = []
    for i in range(1,n-1):
        for j in range(1,n-1):
            row.append(z);col.append(i*n+j);data.append(1)
            row.append(z);col.append(i*n+j+1);data.append(-1)
            row.append(z);col.append((i+1)*n+j+1);data.append(1)
            row.append(z);col.append((i+1)*n+j);data.append(-1)

            z = z+1
    E =sp.coo_matrix((data, (row, col)), shape=(z,N))


    row = range(n);col = range(n);data = [1]*n
    I=sp.coo_matrix((data, (row, col)), shape=(n,N))
    x = cp.Variable(N)
    
    
    if num:
        D=sp.csc_matrix(D)
        prob = cp.Problem(cp.Minimize(1*(cp.sum_squares(C @ x-c[:,0])+10*cp.sum_squares(B @ x-b[:,0]))+2*(cp.sum_squares(A @ x)+20*cp.sum_squares(E @ x))+0.00001*cp.sum_squares(x[n:]-x[:-n])+10*cp.sum_squares(D @ x)),
            [])
        
    else:
        prob = cp.Problem(cp.Minimize(1*(cp.sum_squares(C @ x-c[:,0])+10*cp.sum_squares(B @ x-b[:,0]))+2*(cp.sum_squares(A @ x)+20*cp.sum_squares(E @ x))+0.00001*cp.sum_squares(x[n:]-x[:-n])),
                        []) 

    prob.solve(solver=cp.OSQP,
           verbose=True,
           max_iter = 100, 
           eps_abs = 0.1)
    
    return x.value.reshape(n,n)
    
def opt(boundary,line,line1):
    n = 128
    textline = []
    if line:
        for i in range(len(line)):
            line[i] = line[i].astype(np.float32)
            if len(line[i][::3])<=1:
                continue
            textline.append(line[i]/512*(n-1))
            
    textline1 = []
    if line1:
        for i in range(len(line1)):
            line1[i] = line1[i][:,::-1].astype(np.float32)
            if len(line1[i][::3])<=1:
                continue
            textline1.append(line1[i]/512*(n-1))

    N = n*n

    boundary = (boundary.permute(0,2,3,1).cpu().detach().numpy()+1)/2*(n-1)
    top = boundary[0,0,:,:]
    right = boundary[0,:,-1,:][:,::-1]
    bottom = boundary[0,-1,:,:]
    left = boundary[0,:,0,:][:,::-1]
    
    from multiprocessing import Pool
    pool = Pool(processes=2)

    A = [];B = [] 
    pool.apply_async(grid, args=(left,right,top[:,::-1],bottom[:,::-1],textline,n),callback = A.append)
    pool.apply_async(grid, args=(top,bottom,left[:,::-1],right[:,::-1],textline1,n),callback = B.append) 
    pool.close()
    pool.join()

    v = A[0]
    u = B[0]
    uv = np.stack([u.T,v],-1)
    
    coord_S = torch.Tensor(list(itertools.product(
        np.arange(0, 1 + 0.00001, 1/ (n - 1)),
        np.arange(0, 1 + 0.00001, 1/ (n - 1)),
    )))
    Y, X = coord_S.split(1, dim=1)
    coord_S = torch.cat([X, Y], dim=1)
    coord_S = coord_S.numpy().reshape(n*n,2)#*2-0.5

    uv = uv.reshape(n*n,2)
    transfer_S2T = LinearNDInterpolator(uv,coord_S)
    grid1 = transfer_S2T(coord_S)
    grid1 = 2*grid1-1
    
    return torch.from_numpy(grid1.reshape(n,n,2))