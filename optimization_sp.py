import tensorflow as tf
import os
import time
import math as m
from matplotlib import pyplot as plt
import numpy as np
import json
import scipy
import requests
import IPython.display
import scipy.io
import pandas as pd
import collections
from sklearn.linear_model import LinearRegression
import cupy as cp
import cupyx as cpx
from cupyx.scipy.sparse import linalg as linalg_g
from utils import plot_params,display_seg,plot3d,plot_iso,plot_iso2
# import mcubes
# from skimage.measure import marching_cubes_lewiner

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def sp_cg(A,b,x0=None,M=None,max_iter=None, tol=None):

    if max_iter is None:
        max_iter = 10 * A.shape[0]
    if tol is None:
        tol = 1e-5*tf.linalg.norm(b)

    if x0 is None:
        x0 = tf.zeros_like(b)
    if M is None:
        M = tf.sparse.eye(A.shape[0])
    
    x = tf.reshape(tf.constant(x0, dtype=tf.float32),[-1,1])
    r = b - tf.sparse.sparse_dense_matmul(A,x)
    z =  tf.sparse.sparse_dense_matmul(tf.sparse.transpose(M),r)
    p = tf.constant(z)
    iteration = 0
    while iteration < max_iter and tf.matmul(tf.transpose(r),r)> tol:
        rz_k = tf.matmul(tf.transpose(r),z)
        ap_k = tf.sparse.sparse_dense_matmul(A,p)
        alpha = rz_k / tf.matmul(tf.transpose(p),ap_k)

        x = x + alpha*p
        r = r - alpha * ap_k
        z = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(M),r)
        beta = tf.matmul(tf.transpose(r),z) / rz_k
        p = z + beta * p
        iteration += 1
    # print(iteration)
    return x[:,0]


def lk_H8_np(nu):
    A = np.array([[32, 6, -8, 6, -6, 4, 3, -6, -10, 3, -3, -3, -4, -8], [-48, 0, 0, -24, 24, 0, 0, 0, 12, -12, 0, 12, 12, 12]],dtype = float)
    k = A.T@np.array([[1],[nu]])/144.0
    K1 = np.array([k[0],k[1],k[1],k[2],k[4],k[4],k[1],k[0],k[1],k[3],k[5],k[6],k[1],k[1],k[0],k[3],k[6],k[5],k[2],k[3],k[3],k[0],k[7],k[7],k[4],k[5],k[6],k[7],k[0],k[1],k[4],k[6],k[5],k[7],k[1],k[0]])
    K2 = np.array([k[8],k[7],k[11],k[5],k[3],k[6],k[7],k[8],k[11],k[4],k[2],k[4],k[9],k[9],k[12],k[6],k[3],k[5],k[5],k[4],k[10],k[8],k[1],k[9],k[3],k[2],k[4],k[1],k[8],k[11],k[10],k[3],k[5],k[11],k[9],k[12]])
    K3 = np.array([k[5],k[6],k[3],k[8],k[11],k[7],k[6],k[5],k[3],k[9],k[12],k[9],k[4],k[4],k[2],k[7],k[11],k[8],k[8],k[9],k[1],k[5],k[10],k[4],k[11],k[12],k[9],k[10],k[5],k[3],k[1],k[11],k[8],k[3],k[4],k[2]])
    K4 = np.array([k[13],k[10],k[10],k[12],k[9],k[9],k[10],k[13],k[10],k[11],k[8],k[7],k[10],k[10],k[13],k[11],k[7],k[8],k[12],k[11],k[11],k[13],k[6],k[6],k[9],k[8],k[7],k[6],k[13],k[10],k[9],k[7],k[8],k[6],k[10],k[13]])
    K5 = np.array([k[0],k[1],k[7],k[2],k[4],k[3],k[1],k[0],k[7],k[3],k[5],k[10],k[7],k[7],k[0],k[4],k[10],k[5],k[2],k[3],k[4],k[0],k[7],k[1],k[4],k[5],k[10],k[7],k[0],k[7],k[3],k[10],k[5],k[1],k[7],k[0]])
    K6 = np.array([k[13],k[10],k[6],k[12],k[9],k[11],k[10],k[13],k[6],k[11],k[8],k[1],k[6],k[6],k[13],k[9],k[1],k[8],k[12],k[11],k[9],k[13],k[6],k[10],k[9],k[8],k[1],k[6],k[13],k[6],k[11],k[1],k[8],k[10],k[6],k[13]])
    K1 = np.reshape(K1,[6,6])
    K2 = np.reshape(K2,[6,6])
    K3 = np.reshape(K3,[6,6])
    K4 = np.reshape(K4,[6,6])
    K5 = np.reshape(K5,[6,6])
    K6 = np.reshape(K6,[6,6])
    KE = np.concatenate([np.concatenate([K1, K2, K3, K4],1),np.concatenate([K2.T, K5, K6, K3.T],1),np.concatenate([K3.T, K6, K5.T, K2.T],1),np.concatenate([K4, K3, K2, K1],1)],0)/((nu+1)*(1-2*nu))
    return KE

def brick_stiffnessMatrix(nu):
    D = (1/((1+nu)*(1-2*nu)))*np.array([[1-nu, nu, nu, 0, 0, 0],[ nu, 1-nu, nu, 0, 0, 0],[nu, nu, 1-nu, 0, 0, 0],[ 0, 0, 0, (1-2*nu)/2, 0, 0],[ 0, 0, 0, 0, (1-2*nu)/2, 0],[0, 0, 0, 0, 0, (1-2*nu)/2]])
    B_1=np.array([[-0.044658,0,0,0.044658,0,0,0.16667,0],
    [0,-0.044658,0,0,-0.16667,0,0,0.16667],
    [0,0,-0.044658,0,0,-0.16667,0,0],
    [-0.044658,-0.044658,0,-0.16667,0.044658,0,0.16667,0.16667],
    [0,-0.044658,-0.044658,0,-0.16667,-0.16667,0,-0.62201],
    [-0.044658,0,-0.044658,-0.16667,0,0.044658,-0.62201,0]])
    B_2=np.array([[0,-0.16667,0,0,-0.16667,0,0,0.16667],
    [0,0,0.044658,0,0,-0.16667,0,0],
    [-0.62201,0,0,-0.16667,0,0,0.044658,0],
    [0,0.044658,-0.16667,0,-0.16667,-0.16667,0,-0.62201],
    [0.16667,0,-0.16667,0.044658,0,0.044658,-0.16667,0],
    [0.16667,-0.16667,0,-0.16667,0.044658,0,-0.16667,0.16667]])
    B_3=np.array([[0,0,0.62201,0,0,-0.62201,0,0],
    [-0.62201,0,0,0.62201,0,0,0.16667,0],
    [0,0.16667,0,0,0.62201,0,0,0.16667],
    [0.16667,0,0.62201,0.62201,0,0.16667,-0.62201,0],
    [0.16667,-0.62201,0,0.62201,0.62201,0,0.16667,0.16667],
    [0,0.16667,0.62201,0,0.62201,0.16667,0,-0.62201]])
    B = np.concatenate((B_1,B_2,B_3),axis = 1)
    return B,D

class FEA:

    def __init__(self, nely,nelx,nelz,lele):
        self.nely, self.nelx, self.nelz = nely,nelx,nelz
        self.nele = nelx*nely*nelz
        self.penal = 2.0
        self.lele = lele
    def init_mat(self, E0, Emin, nu):
        self.nu = nu
        self.E0 = E0
        self.Emin = Emin

    def init_bc(self, il, jl, kl, il_F, jl_F,  kl_F, iif, jf, kf):

        nely,nelx,nelz,nele = self.nely, self.nelx,self.nelz,self.nele
        loadnid = kl*(nelx+1)*(nely+1) + il*(nely+1)+(nely+1-jl)
        loaddofx = 3*loadnid-2
        loaddofy = 3*loadnid-1
        loaddofz = 3*loadnid

        loaddofs = np.concatenate([loaddofx,loaddofy,loaddofz],0)
        loaddofs = np.concatenate([np.expand_dims(loaddofs-1,1),np.zeros([il.shape[0]*3,1])],1)
        loaddofv = np.concatenate([il_F,jl_F,kl_F],0)
        ndof = 3*(nelx+1)*(nely+1)*(nelz+1)

        fixednid = kf*(nelx+1)*(nely+1)+iif*(nely+1)+(nely+1-jf)
        fixeddof = np.concatenate([3*fixednid,3*fixednid-1,3*fixednid-2],1)


        self.freedofs = np.setdiff1d(np.linspace(1.0,ndof,ndof), fixeddof.reshape([fixeddof.size]))
        nodegrd = np.transpose(np.reshape(np.linspace(1.0,(nely+1)*(nelx+1),(nely+1)*(nelx+1)),[nelx+1,nely+1]))
        nodeids = np.reshape(np.transpose(nodegrd[0:nely,0:nelx]),[nely*nelx,1])
        nodeidz = np.linspace(0.0,(nelz-1)*(nely+1)*(nelx+1),(nelz))
        nodeids = nodeids*np.ones([1,nodeidz.shape[0]])+np.ones([nely*nelx,1])*nodeidz
        edofVec = 3*np.reshape(np.transpose(nodeids),[np.size(nodeids),1])+1
        self.edofMat = edofVec*np.ones([1,24]) + np.ones([nele,1])*np.array([0,1,2,3*nely+3,3*nely+4,3*nely+5,3*nely,3*nely+1,3*nely+2,-3,-2,-1,
                            3*(nely+1)*(nelx+1),3*(nely+1)*(nelx+1)+1,3*(nely+1)*(nelx+1)+2,3*(nely+1)*(nelx+1)+(3*nely+3),
                            3*(nely+1)*(nelx+1)+(3*nely+4),3*(nely+1)*(nelx+1)+(3*nely+5),3*(nely+1)*(nelx+1)+(3*nely),
                            3*(nely+1)*(nelx+1)+(3*nely+1),3*(nely+1)*(nelx+1)+(3*nely+2),3*(nely+1)*(nelx+1)-3,3*(nely+1)*(nelx+1)-2,
                            3*(nely+1)*(nelx+1)-1],dtype=float)
        self.iK = np.reshape(np.repeat(self.edofMat,24*np.ones([nele],dtype=int),axis = 0),[24*24*nele])
        self.jK = np.reshape(np.repeat(self.edofMat,24*np.ones([24],dtype=int),axis = 1),[24*24*nele])
        self.freedofs = np.reshape(self.freedofs-1,[np.size(self.freedofs)]).astype(int)
        # F = scipy.sparse.coo_matrix((loaddofv, (loaddofs.astype(int)[:,0],loaddofs.astype(int)[:,1])), shape = [ndof,1])
        # self.F_f = F.tocsc()[self.freedofs]
        # self.U_prev = tf.zeros(self.freedofs.shape)
        #send necessary data into gpu
        self.iK_g = cp.array(self.iK)
        self.jK_g = cp.array(self.jK)
        F_g = cpx.scipy.sparse.coo_matrix((cp.array(loaddofv), (cp.array(loaddofs.astype(int)[:,0]),cp.array(loaddofs.astype(int)[:,1]))), shape = (ndof,1))
        self.F_f_g = F_g.tocsc()[self.freedofs]
        self.KE_g = cp.array(self.KE)
        self.freedofs_g = cp.array(self.freedofs)
        self.U_prev_g = cp.zeros(self.freedofs.shape)

    @tf.custom_gradient
    def compliance_cp(self, x):
        xPhys = cp.array(x.numpy())
        start = time.time()
        sK = cp.reshape(self.KE_g,[-1,1])*(self.Emin + cp.power(cp.reshape(cp.transpose(xPhys),[1,-1]),self.penal)*(self.E0-self.Emin))
        sK = cp.reshape(cp.transpose(sK),[24*24*self.nele,1])
        #K = coo_matrix((sK.reshape([-1]), ((jK-1).astype(int), (iK-1).astype(int))))
        K = cpx.scipy.sparse.coo_matrix((sK.reshape([-1]), ((self.jK_g-1).astype(int), (self.iK_g-1).astype(int))))
        K_f = K.tocsc()[self.freedofs] 
        K_f = K_f.transpose()[self.freedofs]
        f_size = self.freedofs.size
        K_fdiag = cp.asarray(K_f[cp.linspace(0,f_size-1,f_size,dtype=int),cp.linspace(0,f_size-1,f_size,dtype=int)])
        #M = coo_matrix((K_fdiag.reshape([f_size]),(np.linspace(0,f_size-1,f_size,dtype=int),np.linspace(0,f_size-1,f_size,dtype=int))))
        #M = cpx.scipy.sparse.coo_matrix((K_fdiag.reshape([f_size]),(cp.linspace(0,f_size-1,f_size,dtype=int),cp.linspace(0,f_size-1,f_size,dtype=int))))
        M =  cpx.scipy.sparse.spdiags(data = 1/K_fdiag,diags = 0, m = K_fdiag.size,n = K_fdiag.size)
        #print ('GPU Assembly took {} sec'.format(time.time()-start))
        start = time.time()
        if self.nele>5000:
            #U_f,_ = linalg_g.cg(K_f,F_f_g.toarray(),x0=U_prev_g,M=M,maxiter = 4000)
            U_f,_ = linalg_g.cg(K_f,self.F_f_g.toarray(),x0=self.U_prev_g,M=M,maxiter = 8000)
            #U_f,_ = linalg_g.cg(K_f,self.F_f_g.toarray(),M=M,maxiter = 8000)
        else:
            U_f = linalg_g.spsolve(K_f,self.F_f_g.toarray())
        self.U_prev_g = U_f
        #U_f = linalg.spsolve(K_f,F_f)
        #print ('GPU Linear solver took {} sec'.format(time.time()-start))
        #U = coo_matrix((U_f, (freedofs,np.zeros(freedofs.size,dtype = int)))).toarray()
        U = cpx.scipy.sparse.coo_matrix((U_f, (self.freedofs_g,cp.zeros(self.freedofs.size,dtype = int)))).toarray()
        Ei = self.Emin + cp.power(xPhys,self.penal)*(self.E0-self.Emin)
        U_e = U[cp.reshape(self.edofMat-1,[-1]).astype(int)].reshape([self.edofMat.shape[0],self.edofMat.shape[1]])
        ce = cp.sum(U_e@self.KE_g*U_e,axis=1)
        Ei_t = cp.reshape(cp.transpose(Ei),[self.nelx*self.nely*self.nelz,1])
        c = tf.reduce_sum(tf.pow(xPhys.get(),self.penal)*tf.convert_to_tensor(ce.get().reshape([self.nely,self.nelx,self.nelz],order='F'),dtype=tf.float32))
        
        def grad(dy):
            dc = -dy*self.penal*tf.pow(xPhys.get(),self.penal-1.0)*tf.convert_to_tensor(ce.get().reshape([self.nely,self.nelx,self.nelz],order='F'),dtype=tf.float32)
            #tf.print('dc:\n',dc)
            return tf.reshape(dc,[1,-1])
        return tf.convert_to_tensor(c,dtype=tf.float32),grad
    def max_disp(self, x):
        xPhys = cp.array(x.numpy())
        start = time.time()
        sK = cp.reshape(self.KE_g,[-1,1])*(self.Emin + cp.power(cp.reshape(cp.transpose(xPhys),[1,-1]),self.penal)*(self.E0-self.Emin))
        sK = cp.reshape(cp.transpose(sK),[24*24*self.nele,1])
        #K = coo_matrix((sK.reshape([-1]), ((jK-1).astype(int), (iK-1).astype(int))))
        K = cpx.scipy.sparse.coo_matrix((sK.reshape([-1]), ((self.jK_g-1).astype(int), (self.iK_g-1).astype(int))))
        K_f = K.tocsc()[self.freedofs] 
        K_f = K_f.transpose()[self.freedofs]
        f_size = self.freedofs.size
        K_fdiag = cp.asarray(K_f[cp.linspace(0,f_size-1,f_size,dtype=int),cp.linspace(0,f_size-1,f_size,dtype=int)])
        #M = coo_matrix((K_fdiag.reshape([f_size]),(np.linspace(0,f_size-1,f_size,dtype=int),np.linspace(0,f_size-1,f_size,dtype=int))))
        #M = cpx.scipy.sparse.coo_matrix((K_fdiag.reshape([f_size]),(cp.linspace(0,f_size-1,f_size,dtype=int),cp.linspace(0,f_size-1,f_size,dtype=int))))
        M =  cpx.scipy.sparse.spdiags(data = 1/K_fdiag,diags = 0, m = K_fdiag.size,n = K_fdiag.size)
        #print ('GPU Assembly took {} sec'.format(time.time()-start))
        start = time.time()
        if self.nele>5000:
            #U_f,_ = linalg_g.cg(K_f,F_f_g.toarray(),x0=U_prev_g,M=M,maxiter = 4000)
            U_f,_ = linalg_g.cg(K_f,self.F_f_g.toarray(),x0=self.U_prev_g,M=M,maxiter = 8000)
            #U_f,_ = linalg_g.cg(K_f,self.F_f_g.toarray(),M=M,maxiter = 8000)
        else:
            U_f = linalg_g.spsolve(K_f,self.F_f_g.toarray())
        # self.U_prev_g = U_f
        #U_f = linalg.spsolve(K_f,F_f)
        #print ('GPU Linear solver took {} sec'.format(time.time()-start))
        #U = coo_matrix((U_f, (freedofs,np.zeros(freedofs.size,dtype = int)))).toarray()
        U = cpx.scipy.sparse.coo_matrix((U_f, (self.freedofs_g,cp.zeros(self.freedofs.size,dtype = int)))).toarray()
        Ei = self.Emin + cp.power(xPhys,self.penal)*(self.E0-self.Emin)
        U_e = U[cp.reshape(self.edofMat-1,[-1]).astype(int)].reshape([self.edofMat.shape[0],self.edofMat.shape[1]])
        U_e = U_e.get()
        # U_e_max = np.max(np.abs(U_e))
        U_e3 = U_e.reshape([-1,8,3])
        U_e3 = np.mean(U_e3,axis=1)
        U_er1 = U_e3.reshape([self.nely,self.nelx,self.nelz,3],order='F')
        U_er = U_er1.reshape([-1,3])
        max_d = np.max(abs(U_er))
        return max_d

    # @tf.custom_gradient
    # def compliance_cp(self, x):
    #     xPhys = x.numpy()
    #     start = time.time()
    #     global penal
    #     #print('xPhys into compliance:\n',xPhys)
    #     sK = np.reshape(self.KE,[np.size(self.KE),1])*(self.Emin + np.power(np.reshape(np.transpose(xPhys),[1,np.size(xPhys)]),self.penal)*(self.E0-self.Emin))
    #     sK = np.reshape(np.transpose(sK),[24*24*self.nele,1])
    #     K = scipy.sparse.coo_matrix((np.reshape(sK,[np.size(sK)]), ((self.jK-1).astype(int), (self.iK-1).astype(int))))
    #     K_f = K.tocsc()[self.freedofs] 
    #     K_f = K_f.transpose()[self.freedofs]
    #     f_size = np.size(self.freedofs)
    #     K_fdiag = np.asarray(K_f[np.linspace(0,f_size-1,f_size,dtype=int),np.linspace(0,f_size-1,f_size,dtype=int)])
    #     M = scipy.sparse.spdiags(data = 1/K_fdiag,diags = 0, m = K_fdiag.size,n = K_fdiag.size)

    #     K_f_coo = K_f.tocoo()
    #     indices_K_f = np.mat([K_f_coo.row, K_f_coo.col]).transpose()
    #     K_f_t = tf.SparseTensor(indices_K_f, tf.cast(K_f_coo.data,dtype=tf.float32), K_f_coo.shape)
    #     M_coo = M.tocoo()
    #     indices_M = np.mat([M_coo.row, M_coo.col]).transpose()
    #     M_t = tf.SparseTensor(indices_M , tf.cast(M_coo.data,dtype=tf.float32), M_coo.shape)
    #     start = time.time()
    #     U_f = sp_cg(K_f_t,tf.cast(self.F_f.toarray(),dtype=tf.float32),x0 = self.U_prev,M=M_t,max_iter = 8000)
    #     self.U_prev = U_f
    #     U = scipy.sparse.coo_matrix((U_f, (self.freedofs,np.zeros(self.freedofs.size,dtype = int)))).toarray()
    #     Ei = self.Emin + np.power(xPhys,self.penal)*(self.E0-self.Emin)
    #     U_e = U[np.reshape(self.edofMat-1,[np.size(self.edofMat)]).astype(int)].reshape([self.edofMat.shape[0],self.edofMat.shape[1]])
    #     # U_e3 = U_e.reshape([-1,8,3])
    #     # U_e3 = np.mean(U_e3,axis=1)
    #     # self.max_d = np.max(abs(U_e3))
    #     ce = np.sum(U_e@self.KE*U_e,axis=1)
    #     c = tf.reduce_sum(tf.pow(xPhys,self.penal)*tf.convert_to_tensor(ce.reshape([self.nely,self.nelx,self.nelz],order='F'),dtype=tf.float32))
    #     def grad(dy):
    #         dc = -dy*self.penal*tf.pow(xPhys,self.penal-1.0)*tf.convert_to_tensor(ce.reshape([self.nely,self.nelx,self.nelz],order='F'),dtype=tf.float32)
    #         return dc
    #     return tf.convert_to_tensor(c,dtype=tf.float32),grad

    # def stress_calc(self, x):
    #     B,D=brick_stiffnessMatrix(self.nu)
    #     MISES=np.zeros((self.nele,1))
    #     S=np.zeros((self.nele,6))
    #     for i in range(self.nele):
    #         edof_index = self.edofMat[i,:].astype(int) - 1
    #         temp=np.dot(np.reshape((x[i,0]**self.q),(1,1)),np.transpose(np.dot(np.dot(D,B),np.reshape(self.U[edof_index],(self.edofMat.shape[1],1)))))
    #         S[i,:]=temp
    #         MISES[i,0]=np.sqrt(0.5*((temp[0,0]-temp[0,1])**2+(temp[0,0]-temp[0,2])**2+(temp[0,1]-temp[0,2])**2+6*sum(temp[0,3:5]**2)))

    #     pnorm=(np.sum(MISES**self.pnorm_agg))**(1/self.pnorm_agg)
    #     return pnorm,MISES


    # def max_disp(self, x):
    #     xPhys = x.numpy()
    #     sK = np.reshape(self.KE,[np.size(self.KE),1])*(self.Emin + np.power(np.reshape(np.transpose(xPhys),[1,np.size(xPhys)]),self.penal)*(self.E0-self.Emin))
    #     sK = np.reshape(np.transpose(sK),[24*24*self.nele,1])
    #     K = scipy.sparse.coo_matrix((np.reshape(sK,[np.size(sK)]), ((self.jK-1).astype(int), (self.iK-1).astype(int))))
    #     K_f = K.tocsc()[self.freedofs] 
    #     K_f = K_f.transpose()[self.freedofs]
    #     f_size = np.size(self.freedofs)
    #     K_fdiag = np.asarray(K_f[np.linspace(0,f_size-1,f_size,dtype=int),np.linspace(0,f_size-1,f_size,dtype=int)])
    #     M = scipy.sparse.spdiags(data = 1/K_fdiag,diags = 0, m = K_fdiag.size,n = K_fdiag.size)
    #     K_f_coo = K_f.tocoo()
    #     indices_K_f = np.mat([K_f_coo.row, K_f_coo.col]).transpose()
    #     K_f_t = tf.SparseTensor(indices_K_f, tf.cast(K_f_coo.data,dtype=tf.float32), K_f_coo.shape)
    #     M_coo = M.tocoo()
    #     indices_M = np.mat([M_coo.row, M_coo.col]).transpose()
    #     M_t = tf.SparseTensor(indices_M , tf.cast(M_coo.data,dtype=tf.float32), M_coo.shape)
    #     U_f = sp_cg(K_f_t,tf.cast(self.F_f.toarray(),dtype=tf.float32),x0 = self.U_prev,M=M_t,max_iter = 8000)
    #     # self.U_prev = U_f
    #     U = scipy.sparse.coo_matrix((U_f, (self.freedofs,np.zeros(self.freedofs.size,dtype = int)))).toarray()
    #     U_e = U[np.reshape(self.edofMat-1,[np.size(self.edofMat)]).astype(int)].reshape([self.edofMat.shape[0],self.edofMat.shape[1]])
    #     U_e3 = U_e.reshape([-1,8,3])
    #     U_e3 = np.mean(U_e3,axis=1)
    #     U_er1 = U_e3.reshape([self.nely,self.nelx,self.nelz,3],order='F')
    #     U_er = U_er1.reshape([-1,3])
    #     max_d = np.max(abs(U_er))
    #     return max_d
    
#functions for optimization 
def r3d_tf(angle):
    sx,sy,sz = (tf.constant(0.0,dtype=tf.float32),tf.math.sin(angle[0,0]), tf.math.sin(angle[0,1]))
    cx,cy,cz = (tf.constant(1.0,dtype=tf.float32),tf.math.cos(angle[0,0]), tf.math.cos(angle[0,1]))
    m00 = cy * cz
    m01 = (sx * sy * cz) - (cx * sz)
    m02 = (cx * sy * cz) + (sx * sz)
    m10 = cy * sz
    m11 = (sx * sy * sz) + (cx * cz)
    m12 = (cx * sy * sz) - (sx * cz)
    m20 = -sy
    m21 = sx * cy
    m22 = cx * cy
    matrix = tf.stack((m00, m10, m20,m01, m11, m21,m02, m12, m22),axis=-1)
    return tf.transpose(tf.reshape(matrix,[3,3]))
    
def r3d_tf2(angle):
    return r3d_tf(angle*tf.constant([[1.0,0.0]],dtype=tf.float32))@r3d_tf(angle*tf.constant([[0.0,1.0]],dtype=tf.float32),)

def euler2vec(rot): # rot X, Z
    vecX = tf.math.sin(rot[:,1:2])
    vecZ = -tf.math.cos(0.5*m.pi - rot[:,0:1]) * tf.math.cos(rot[:,1:2])
    vecY = tf.math.sin(0.5*m.pi - rot[:,0:1]) * tf.math.cos(rot[:,1:2])
    return tf.concat([vecY,vecX,vecZ],axis = 1)


class nnopt:


    
    def __init__(self,nely,nelx,nelz,vf,lele,directory_path):
        #Problem setup
        self.nely, self.nelx, self.nelz = nely,nelx,nelz
        self.nele = self.nelz*self.nely*self.nelx 
        self.nelm = max(nelx,nely,nelz)
        self.volfrac = vf
        #self.part_length_m = part_length_m #part length max in meters
        self.lele = lele
        self.fea = FEA(self.nely, self.nelx, self.nelz,self.lele)

        self.xPhys_final_np = np.ones([nely,nelx,nelz])*vf
        self.m_dir = [0,0,0,0,0,0] #3 axis subtractive in the order of y+, y-, x+, x-, z+, z-
        self.c_dir = [0,0,0]  #cutting direction in y,x,z
        self.a_dir = [0,0,0,0,0,0] # fixed axis metal additive/FDM in the order of y+, y-, x+, x-, z+, z-
        self.soa = 45.0 #support overhang angle, 90:support everything, 0: no support
        SV_init = 0.0
        self.SV_init = 0.0
        self.SV_max =  10.0
        self.SV_delta = (self.SV_max - self.SV_init)/100.0
        self.SV_coeff = SV_init

        self.SV_init2 = 0.0
        self.SV_max2 =  100.0
        self.SV_delta2 = (self.SV_max2 - self.SV_init2)/200.0
        self.SV_coeff2 = self.SV_init2

        self.SV_initam = 0.0
        self.SV_maxam =  100.0
        self.SV_deltaam = (self.SV_maxam - self.SV_initam)/200.0
        self.SV_coeffam = self.SV_initam

        self.directory_path = directory_path

        self.unrestricted = True #unrestricted topology optimization
        #neural network setup
        self.total_epoch = 0
        def dl_linspace(nele,SS):
            return np.linspace(-(nele-1)/(2*self.nelm ),(nele-1)/(2*self.nelm ),nele*SS)
        c_y, c_x, c_z=np.meshgrid(dl_linspace(nely,1),dl_linspace(nelx,1),dl_linspace(nelz,1),indexing='ij')
        dlX = np.stack((c_y.reshape([-1]),c_x.reshape([-1]),c_z.reshape([-1])),axis = 1).reshape([-1,3])
        self.dlX = tf.convert_to_tensor(dlX,dtype=tf.float32)*tf.constant([[-1.0,1.0,1.0]])
        c_y, c_x, c_z=np.meshgrid(dl_linspace(nely,2),dl_linspace(nelx,2),dl_linspace(nelz,2),indexing='ij')
        dlXSS = np.stack((c_y.reshape([-1]),c_x.reshape([-1]),c_z.reshape([-1])),axis = 1).reshape([-1,3])
        self.dlXSS = tf.convert_to_tensor(dlXSS,dtype=tf.float32)*tf.constant([[-1.0,1.0,1.0]])

        low_band = 0.0
        high_band = 25
        c_y, c_x, c_z=np.meshgrid(np.linspace([-high_band,low_band],[-low_band,high_band],6).reshape([-1]),
                                                    np.linspace([-high_band,low_band],[-low_band,high_band],6).reshape([-1]),
                                                    np.linspace([-high_band,low_band],[-low_band,high_band],6).reshape([-1]),indexing='ij')
        dlInit = np.stack((c_y.reshape([-1]),c_x.reshape([-1]),c_z.reshape([-1])),axis = 0)
        self.kernel = tf.Variable(dlInit,trainable=True,dtype=tf.float32)
        self.weights1 = tf.Variable(tf.zeros([dlInit.shape[1],1]) + 0.00001,trainable=True)
        self.bias1 = tf.Variable(tf.ones([1,dlInit.shape[1]]),trainable=True,dtype=tf.float32)
        self.bias2 = tf.Variable(tf.zeros([1,dlInit.shape[1]]),trainable=True,dtype=tf.float32)

        self.to_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
        self.to_optimizerEDM = tf.keras.optimizers.Adam(learning_rate=0.0005)

        self.lbd = 1.0 #augmented lagrangian 

        self.penal_init = 2.0
        self.penal_max = 4.0
        self.penal_delta = 0.01

        self.alpha_init = 1
        self.alpha_max = 10
        self.alpha_delta = 0.05

        self.log_sv = []
        self.log_compliance = []
        self.log_vf = []
        self.log_time = []
        self.log_cost = []

        self.debug = False

        drho_shell = np.zeros((self.nely,self.nelx,self.nelz,3))
        drho_shell[0,:,:,:] += np.array([-1.0,0,0])
        drho_shell[-1,:,:,:] += np.array([1.0,0,0])
        drho_shell[:,0,:,:] += np.array([0,1.0,0])
        drho_shell[:,-1,:,:] += np.array([0,-1.0,0])
        drho_shell[:,:,0,:] += np.array([0,0,1.0])
        drho_shell[:,:,-1,:] += np.array([0,0,-1.0])
        drho_shell = drho_shell.reshape([-1,3])
        drho_shell = drho_shell / np.sqrt(np.sum(drho_shell**2.0,axis=1,keepdims=True)+0.0001 ) 
        self.drho_shell = drho_shell

        # print(os.listdir(os.path.dirname(directory_path)))
        # if 'passive.npy' in os.listdir(os.path.dirname(directory_path)):
        #     print('Passive Present')
        #     self.passive = np.load(os.path.join(os.path.dirname(directory_path),'passive.npy'))
        # self.dlX_passive = tf.boolean_mask(self.dlX,tf.convert_to_tensor(self.passive.reshape([-1]), dtype=tf.bool))

        # #Active Regions
        # xPhys_load = np.zeros((16,40,40))
        # for ix in range(0,41):
        #     for iz in range(0,41):
        #         if (ix-19.5)**2 + (iz-19.5)**2 >= 19**2 and (ix-19.5)**2 + (iz-19.5)**2 <= 20**2: 
        #             xPhys_load[0,ix,iz] = 1.0
        # xPhys_fixed = np.zeros((16,40,40))
        # xPhys_fixed[15,16,16] = 1.0
        # xPhys_fixed[15,16,23] = 1.0
        # xPhys_fixed[15,23,16] = 1.0
        # xPhys_fixed[15,23,23] = 1.0
        # xPhys_active = xPhys_load + xPhys_fixed
        # self.active = xPhys_active
        # p_index = np.where(self.active.reshape([-1]))[0]
        # self.dlX_active = tf.convert_to_tensor(self.dlX.numpy()[p_index,:])
        # self.nele_active = np.sum(self.active)
        # print('self.nele_active',self.nele_active)

        # #Passive Regions
        # xPhys_passive = np.zeros((16,40,40))
        # xPhys_passive[5:,25:40,25:40] = 1.0
        # self.passive = xPhys_passive
        # p_index = np.where(self.passive.reshape([-1]))[0]
        # self.dlX_passive = tf.convert_to_tensor(self.dlX.numpy()[p_index,:])




    #initialize the FEA solver
    def init_mat(self, E0, Emin, nu,mat_name):
        self.mat_name = mat_name
        self.fea.init_mat(E0, Emin, nu)

    def init_bc(self, il, jl, kl, il_F, jl_F,  kl_F, iif, jf, kf):
        self.fea.init_bc(il, jl, kl, il_F, jl_F,  kl_F, iif, jf, kf)

    def init_3axis(self, m_dir):
        self.m_dir = m_dir
        self.unrestricted = False

    def init_2axis(self, c_dir):
        self.c_dir = c_dir
        self.unrestricted = False
        if c_dir[0]:
            self.dlX= self.dlX*tf.constant([[0.0,1.0,1.0]])
            self.dlXSS= self.dlXSS*tf.constant([[0.0,1.0,1.0]])
        if c_dir[1]:
            self.dlX = self.dlX*tf.constant([[1.0,0.0,1.0]])
            self.dlXSS = self.dlXSS*tf.constant([[1.0,0.0,1.0]])
        if c_dir[2]:
            self.dlX = self.dlX*tf.constant([[1.0,1.0,0.0]])   
            self.dlXSS = self.dlXSS*tf.constant([[1.0,1.0,0.0]])   
    
    def init_m_additive(self, a_dir, soa = 45.0):
        self.a_dir = a_dir
        self.soa = soa
        self.unrestricted = False

    def steep_sigmoid(self,x):
        return 1/(1+tf.exp(-10*x))

    def rbnn(self, coords):
        scale = 1.0
        layer1 = tf.cos(tf.matmul(coords, (1.0 / scale) *self.kernel) + self.bias1)
        rho = tf.nn.sigmoid(tf.matmul(layer1+ self.bias2, self.weights1) + self.offset )
        return rho            
    
    def drbnn(self, coords):
        with tf.GradientTape(persistent=True)  as g:
            g.watch(coords)
            xPhys = self.rbnn(coords)
        return g.gradient(xPhys,coords)       

    #3 axis machining 
    def apply_m_dir(self, axis, reverse):
        xPhys = tf.reshape( self.rbnn(self.dlX) ,[self.nely,self.nelx,self.nelz] )
        P_c = tf.math.cumsum(xPhys,axis = axis,reverse = reverse)   
        P_c_f = 1.0/(1+tf.exp(-10.0*(P_c-0.5))) #change filtering to 0.5 for direct cumsum on xPhys, 2.0 for P_alpha
        sv = P_c_f*(1.0-xPhys)     
        return sv

    def m_loss(self):
        sv = tf.ones([self.nely,self.nelx,self.nelz],dtype=tf.float32)
        if self.m_dir[0]: #y+ direction
            sv = sv * self.apply_m_dir(0, True)
        if self.m_dir[1]: #y- direction
            sv = sv * self.apply_m_dir( 0, False)
        if self.m_dir[2]: #x+ direction
            sv = sv * self.apply_m_dir(1, False)
        if self.m_dir[3]: #x- direction
            sv = sv * self.apply_m_dir(1, True)
        if self.m_dir[4]: #z+ direction
            sv = sv * self.apply_m_dir(2, False)
        if self.m_dir[5]: #z- direction
            sv = sv * self.apply_m_dir(2, True)
        return sv



    #metal additive / FDM
    def overhang(self, rx,rz,axis, reverse):
        #angle = angle_model(tf.ones([1,1]),training=True)
        angle = tf.constant([[rx, rz]])
        rot_coord = self.dlX@r3d_tf2(angle) #yxz
        xPhys = self.rbnn(self.dlX)
        x_heavy = 1.0/(1+tf.exp(-2.0*10*(xPhys-0.3)))
        drho_dcoord = self.drbnn(self.dlX) + tf.convert_to_tensor(self.drho_shell,dtype = tf.float32)*x_heavy* (self.nelm*1.0)
        dr_dcn = drho_dcoord / tf.sqrt(tf.reduce_sum(drho_dcoord**2.0,axis=1,keepdims=True)+0.01 )
        b = euler2vec(angle)
        cos_alpha_n = tf.reshape(tf.tensordot(b,tf.transpose(dr_dcn),axes=1),[self.nely,self.nelx,self.nelz])
        cos_alpha_over = cos_alpha_n  - tf.math.cos(self.soa*0.0174533)
        h_cos_alpha = 1.0/(1+tf.exp(-2.0*10*(cos_alpha_over+0.0)))
        P_alpha = h_cos_alpha *cos_alpha_n
        xPhys = tf.reshape( xPhys ,[self.nely,self.nelx,self.nelz] )
        P_c = tf.math.cumsum(P_alpha,axis = axis,reverse = reverse)   
        P_c_f = 1.0/(1+tf.exp(-10.0*(P_c-2.0))) #change filtering to 0.5 for direct cumsum on xPhys, 2.0 for P_alpha
        sv = P_c_f*(1.0-xPhys)  
        sv = P_c_f*(1.0-tf.reshape( x_heavy ,[self.nely,self.nelx,self.nelz] ))   
        return sv

    def sv_loss(self):
        sv = tf.ones([self.nely,self.nelx,self.nelz],dtype=tf.float32)
        if self.a_dir[0]: #y+ direction
            sv = sv * self.overhang(0.0, 0.0, 0, False)
        if self.a_dir[1]: #y- direction
            sv = sv * self.overhang(m.pi, 0, 0, True) 
        if self.a_dir[2]: #x+ direction
            sv = sv * self.overhang(0, 0.5*m.pi, 1, True) 
        if self.a_dir[3]: #x- direction
            sv = sv * self.overhang(0, -0.5*m.pi, 1, False)
        if self.a_dir[4]: #z+ direction
            sv = sv * self.overhang(-0.5*m.pi, 0, 2, True) 
        if self.a_dir[5]: #z- direction
            sv = sv * self.overhang(0.5*m.pi, 0, 2, False)
        return sv

    def edm_area(self):
        with tf.GradientTape(persistent=True)  as g:
            g.watch(self.dlX)
            xPhys = self.rbnn(self.dlX)
        drho_dcoord = g.gradient(xPhys,self.dlX)
        dr_dcn = tf.sqrt(tf.reduce_sum(drho_dcoord**2.0,axis=1,keepdims=True)+0.01 )

        dr_dcn = tf.reshape(dr_dcn,[self.nely,self.nelx,self.nelz])

        drho_f = 1.0 / (1 + tf.exp(-dr_dcn+5.0))

        edm_area = tf.reduce_sum(drho_f)
        return edm_area
           

    def time_cost_eval(self,guess_vf=None):
        if guess_vf:
            xPhys = tf.ones([self.nele,1])*guess_vf
        else:
            xPhys = self.rbnn(self.dlX)
        if sum(self.c_dir) >0: #edm
            mat_cost = self.nele *self.lele**3 * self.m_density * self.c_mat
            if guess_vf:
                cut_area = self.nele *self.lele**2 * (1.0-guess_vf) 
            else:
                cut_area = self.edm_area() * self.lele **2
            t_cut = cut_area / self.q_cut
            t_total = self.edm_t_setup + t_cut + self.edm_t_inspection +self.edm_t_polish
            c_total = self.edm_c_setup + t_cut*self.edm_m_cost_per_min + self.edm_c_polish + self.edm_c_inspection + mat_cost
            return t_total, c_total

        elif sum(self.m_dir) >0: #subtractive
            mat_cost = self.nelz*self.nely*self.nelx *self.lele**3 * self.m_density * self.c_mat

            m_time = (self.nele - tf.reduce_sum(xPhys)) *self.lele**3 / self.m_removal_rate
            t_fixture = sum(self.m_dir) * self.sub_t_opt_setup
            c_fixture = sum(self.m_dir) * self.sub_c_opt_setup

            t_total = self.sub_t_setup + t_fixture + m_time + self.sub_t_polish + self.sub_t_inspection
            c_total = self.sub_c_setup + c_fixture + m_time*self.sub_m_cost_per_min + self.sub_c_polish + self.sub_c_inspection + mat_cost

            return t_total, c_total


        elif sum(self.a_dir)>0: #additive
            if guess_vf:
                sv_total = 0.1*self.nele * guess_vf
            else:
                sv_total = self.sv_loss()
            sv_density = 0.3
            sv_mass = tf.reduce_sum(sv_total)*self.lele**3 * sv_density * self.m_density 
            mat_cost = (tf.reduce_sum(xPhys) *self.lele**3 * self.m_density + sv_mass) * self.c_mat_powder
            part_mass = tf.reduce_sum(xPhys) *self.lele**3 * self.m_density
            total_mass = sv_mass + part_mass
            print_time = total_mass / self.am_print_rate 
            t_total = self.add_t_setup + print_time + self.add_t_support + self.add_t_inspection 
            c_total =  self.add_c_setup + print_time*self.add_m_cost_per_min + self.add_c_support + self.add_c_inspection + mat_cost
            return t_total, c_total             

        else:
            mat_cost = tf.reduce_sum(xPhys) *self.lele**3 * self.m_density * self.c_mat
            return 0.0,mat_cost
        
    def normalize_time_cost(self):
        self.t_norm, self.c_norm = self.time_cost_eval(guess_vf=0.5)

    def model_loss(self, epoch):
        xPhys = self.rbnn(self.dlX)
        penal = min(self.penal_init + self.penal_delta * epoch,self.penal_max)
        alpha = min(self.alpha_init + self.alpha_delta * epoch, self.alpha_max)
        self.fea.penal = penal
        c = self.fea.compliance_cp(tf.reshape(xPhys,[self.nely,self.nelx,self.nelz]))
        vf = tf.math.reduce_mean(xPhys)
        t_total, c_total = self.time_cost_eval()
        part_mass = tf.reduce_sum(xPhys) *self.lele**3 * self.m_density

        # passive_loss = tf.reduce_mean(self.rbnn(self.dlX_passive))
        # active_loss = tf.reduce_mean(tf.abs(self.rbnn(self.dlX_active) - 1.0))
        # print('active loss: ',active_loss)
        # print('passive loss: ',passive_loss)
        if epoch > 0:
            self.SV_coeff = max(min(self.SV_coeff + self.SV_delta, self.SV_max),self.SV_init)
        else:
            self.SV_coeff = 0.0

        if epoch < 100:
            self.SV_coeff2 = max(min(self.SV_coeff2 + self.SV_delta2, self.SV_max2),self.SV_init2)
        else:
            self.SV_coeff2 =  min(self.SV_coeff2 + (epoch/100)**3,100)

        if epoch < 100:
            self.SV_coeffam = max(min(self.SV_coeffam + self.SV_deltaam, self.SV_maxam),self.SV_initam)
            # self.SV_coeffam = 10
        else:
            self.SV_coeffam =  min(self.SV_coeffam + (epoch/100)**3,100)
        if self.unrestricted:
            loss_height = 0.0
            loss = c/self.c_0 + alpha*(vf/self.volfrac-1.0)**2

        elif sum(self.c_dir) >0: #edm
            loss_height = tf.math.reduce_mean(self.edm_area())
            # loss = c/self.c_0 + self.SV_coeffam*(tf.maximum(0,(part_mass/self.mass_con-1.0)))**2 + self.SV_coeffam*(tf.maximum(0,(c_total/self.cost_con-1.0)))**2 + self.SV_coeffam*(tf.maximum(0,(t_total/self.time_con-1.0)))**2 + self.SV_coeff2*(passive_loss)
            loss = c/self.c_0 + self.SV_coeffam*(tf.maximum(0,(part_mass/self.mass_con-1.0)))**2 + self.SV_coeffam*(tf.maximum(0,(c_total/self.cost_con-1.0)))**2 + self.SV_coeffam*(tf.maximum(0,(t_total/self.time_con-1.0)))**2

        elif sum(self.m_dir) >0: #subtractive
            loss_height = tf.math.reduce_mean(self.m_loss())
            # loss = c/self.c_0 + self.SV_coeff*loss_height**2 + self.lbd*loss_height + self.SV_coeff2*(tf.maximum(0,(part_mass/self.mass_con-1.0)))**2 + self.SV_coeff2*(tf.maximum(0,(c_total/self.cost_con-1.0)))**2 + self.SV_coeff2*(tf.maximum(0,(t_total/self.time_con-1.0)))**2 + self.SV_coeff2*(passive_loss)
            loss = c/self.c_0 + self.SV_coeff*loss_height**2 + self.lbd*loss_height + self.SV_coeff2*(tf.maximum(0,(part_mass/self.mass_con-1.0)))**2 + self.SV_coeff2*(tf.maximum(0,(c_total/self.cost_con-1.0)))**2 + self.SV_coeff2*(tf.maximum(0,(t_total/self.time_con-1.0)))**2
            self.lbd = self.lbd + self.SV_coeff*loss_height

        elif sum(self.a_dir)>0: #additive
            loss_height = tf.math.reduce_mean(self.sv_loss())
            if epoch<100:
                loss = c/self.c_0 + self.SV_coeffam*(tf.maximum(0,(vf/self.init_vf-1.0)))**2
                # loss = c/self.c_0 + self.SV_coeffam*(tf.maximum(0,(vf/self.init_vf-1.0)))**2 + self.SV_coeffam*(active_loss)
                # loss = c/self.c_0 + self.SV_coeffam*(tf.maximum(0,(vf/self.init_vf-1.0)))**2 + self.SV_coeffam*(passive_loss)
            else:    
                # loss = c/self.c_0 + self.SV_coeffam*(tf.maximum(0,(part_mass/self.mass_con-1.0)))**2 + self.SV_coeffam*(tf.maximum(0,(c_total/self.cost_con-1.0)))**2 + self.SV_coeffam*(tf.maximum(0,(t_total/self.time_con-1.0)))**2 + self.SV_coeffam*(passive_loss)
                # loss = c/self.c_0 + self.SV_coeffam*(tf.maximum(0,(part_mass/self.mass_con-1.0)))**2 + self.SV_coeffam*(tf.maximum(0,(c_total/self.cost_con-1.0)))**2 + self.SV_coeffam*(tf.maximum(0,(t_total/self.time_con-1.0)))**2 + self.SV_coeffam*(active_loss)
                loss = c/self.c_0 + self.SV_coeffam*(tf.maximum(0,(part_mass/self.mass_con-1.0)))**2 + self.SV_coeffam*(tf.maximum(0,(c_total/self.cost_con-1.0)))**2 + self.SV_coeffam*(tf.maximum(0,(t_total/self.time_con-1.0)))**2

        if self.debug:
            tf.print('Epoch:',epoch,',Penal:',penal,',Alpha:',alpha,',Lambda SV:',self.SV_coeff, ',Lambda:', self.lbd)
            tf.print('Compliance:',c, ", VF:", vf, ', Total Loss:',loss, ', SV loss:', loss_height)
            tf.print('cost:',c_total, ", time:", t_total)

        if epoch < 0: 
            loss = 1.0*c/self.c_0+alpha*(vf/self.volfrac-1.0)**2 

        self.log_sv.append(loss_height)
        self.log_compliance.append(c)
        self.log_vf.append(vf)
        self.log_time.append(t_total)
        self.log_cost.append(c_total)
        self.xPhys_final_np = xPhys.numpy().reshape([self.nely,self.nelx,self.nelz])
        return loss

    def train_step(self, epoch):
        with tf.GradientTape(persistent=True) as model_tape:
            loss = self.model_loss(epoch)
        param = [self.weights1,self.kernel,self.bias1,self.bias2]
        grad = model_tape.gradient(loss,param)
        grad = [tf.clip_by_norm(g, 0.1) for g in grad]
        if sum(self.c_dir) >0:
            self.to_optimizerEDM.apply_gradients(zip(grad, param))
        else:
            self.to_optimizer.apply_gradients(zip(grad, param))

    def fit(self, epochs):
        for epoch in range(epochs):
            self.total_epoch = self.total_epoch + 1
            self.train_step(self.total_epoch)
            print("Epoch: ", self.total_epoch)    
            if epoch%50 == 49 and epoch>20:
                IPython.display.clear_output(wait=True)
                plot_params(self)
                xPhys = tf.reshape( self.rbnn(self.dlX) ,[self.nely,self.nelx,self.nelz] ).numpy()
                try:
                    plot_iso(self,xPhys,0.2)
                    plot_iso2(xPhys,0.2,210,120)
                    plot_iso2(xPhys,0.2,-90,0)
                    display_seg(self,xPhys)
                except:
                    print('Failed Rendering')

    def display_result_summary(self):
        xPhys = tf.reshape( self.rbnn(self.dlX) ,[self.nely,self.nelx,self.nelz] ).numpy()
        t_total, c_total = self.time_cost_eval()
        self.fea.penal = 3.0
        comp = float((self.fea.compliance_cp(tf.cast(tf.reshape(xPhys,[self.nely,self.nelx,self.nelz]),dtype=tf.float32))*self.fea.E0).numpy())
        # max_d = float(self.fea.max_disp(tf.reshape(xPhys,[self.nely,self.nelx,self.nelz]))*1000)
        mass = tf.reduce_sum(xPhys) *self.lele**3 * self.m_density
        try:
            plot_iso(self,xPhys,0.2)
        except:
            print('Failed Rendering')
        print("## Result Summary")
        print("VF: ",tf.reduce_mean(xPhys))
        print("Mass: {:.6f} [kg]".format(mass))
        print("Compliance: {:2e}".format(comp))
        print("Time: {:.2f} [min]".format(t_total))
        print("Cost: {:.2f} [$]".format(c_total))
        # print("Max Displacement: {:.2f}[mm]".format(max_d))

    def cutoff_calc(self,xPhys):
        cutoff_list = list(np.linspace(0.1,0.9,50))
        cutoff_eval_list = np.array([(np.sum(1.0*(xPhys>i)) - np.sum(xPhys)) for i in cutoff_list])
        cutoff_ind = min(len(cutoff_list)-1,int((max((np.array((np.where(cutoff_eval_list>0))).reshape(-1)).tolist()))))
        cutoff = cutoff_list[int(cutoff_ind)]
        return cutoff

    def m_sv(self,index):
        if index == 0: #y+ direction
            return self.apply_m_dir(0, True)
        if index == 1: #y- direction
            return self.apply_m_dir( 0, False)
        if index == 2: #x+ direction
            return self.apply_m_dir(1, False)
        if index == 3: #x- direction
            return self.apply_m_dir(1, True)
        if index == 4: #z+ direction
            return self.apply_m_dir(2, False)
        if index == 5: #z- direction
            return self.apply_m_dir(2, True)

    def save_m_process_plan(self, guess=None):
        if guess:
            self.probe_dict['Manufacturing Method'].append('Subtractive '+str(self.orientation[:]))
            self.probe_dict['Material'].append(self.mat_name)
            self.probe_dict['Probe'].append(guess)
            xPhys = np.ones([self.nely,self.nelx,self.nelz],dtype = np.float32) * guess
            self.mass_guess = float(np.sum(xPhys) *self.lele**3 * self.m_density* 1000.0)
            self.comp_guess = float((self.fea.compliance_cp(tf.convert_to_tensor(tf.reshape(xPhys,[self.nely,self.nelx,self.nelz]),dtype=tf.float32))*self.fea.E0).numpy())
            self.time_guess,self.cost_guess = self.time_cost_eval(guess_vf=guess)
            self.probe_dict['Mass (g)'].append(self.mass_guess)
            self.probe_dict['Compliance (Nm)'].append(self.comp_guess)
            self.probe_dict['Nominal Time (min)'].append(float(self.time_guess.numpy()))
            self.probe_dict['Nominal Cost ($)'].append(float(self.cost_guess.numpy()))
            c = self.comp_guess
        else:
            xPhys = self.rbnn(self.dlX).numpy().reshape([self.nely,self.nelx,self.nelz])
            cutoff = self.cutoff_calc(xPhys)
            xPhysc = tf.convert_to_tensor(1.0*(xPhys>cutoff),dtype = tf.float32)
            self.fea.penal = 3.0
            c = float((self.fea.compliance_cp(tf.cast(tf.reshape(xPhysc,[self.nely,self.nelx,self.nelz]),dtype=tf.float32))*self.fea.E0).numpy())
        sv = tf.ones([self.nely,self.nelx,self.nelz],dtype=tf.float32)
        xPhysf = (xPhys>0.4).astype(float)*0.5
        opt_num = 0
        sv_prev = 1.0 - xPhys
        
        self.p_plans[0]["ProcessPlans"].extend([{
            "Name":"3-Axis-Subtractive-1",
            "Part": "Bracket",
            "Link": self.directory_path,
            "Material": self.mat_name,
            "ManufacturingMethod":"3-Axis-Subtractive",
            "BulkGrams": float(self.nele* self.lele**3 * self.m_density * 1000.0),
            "NetGrams":float(np.sum(xPhys) *self.lele**3 * self.m_density* 1000.0),
            "ScrapGrams": float((self.nele - np.sum(xPhys)) * self.lele**3 * self.m_density * 1000.0),
            "Width":float(self.nelx*self.lele*1000.0),
            "Height":float(self.nely*self.lele*1000.0),
            "Depth":float(self.nelz*self.lele*1000.0),
            "Compliance":c,
            "Volume":float(np.sum(xPhys) *self.lele**3 * 1e9),
            "Task_Sequence":[{
                "Name":"3-Axis-machine-setup",
                "RequiredCapability":"3-Axis-Machining",
                "NominalDuration":float(self.sub_t_setup*60.0),
                "NominalCost":float(self.sub_c_setup)
                }]
            }])

        for index in range(6):
            if self.m_dir[index]:
                opt_num+=1
                if guess is None:
                    sv = sv * self.m_sv(index)
                    svp = (sv.numpy()>0.6).astype(float)
                    
                    m_remove =  sv_prev - sv
                    sv_prev = sv
                    m_time = tf.reduce_sum(m_remove)*self.lele**3 / self.m_removal_rate
                    m_p = (m_remove.numpy()>0.6).astype(float)*0.31
                    plot3d(xPhysf+m_p, save = True)
                    plt.savefig( os.path.join(self.directory_path,"opt_{}_m_removal".format(opt_num)), transparent=True)
                    plot3d(xPhysf+svp, save = True)            
                    plt.savefig(os.path.join(self.directory_path,"opt_{}_m_left".format(opt_num)), transparent=True)
                else:
                    m_time = (self.nele - tf.reduce_sum(xPhys)) *self.lele**3 / self.m_removal_rate
                    m_time = m_time / sum(self.m_dir)
                self.p_plans[0]["ProcessPlans"][0]["Task_Sequence"].extend([{
                    "Name":"3-Axis-fixture-{}".format(opt_num),
                    "RequiredCapability":"3-Axis-Machining",
                    "NominalDuration":float(self.sub_t_opt_setup*60.0),
                    "NominalCost":float(self.sub_c_opt_setup)
                },
                {
                    "Name":"3-Axis-machining-{}".format(opt_num),
                    "RequiredCapability":"3-Axis-Machining",
                    "NominalDuration":float(m_time*60.0),
                    "NominalCost":float(m_time*self.sub_m_cost_per_min )                   
                }])

        self.p_plans[0]["ProcessPlans"][0]["Task_Sequence"].extend([{
            "Name":"Polishing",
            "RequiredCapability":"Buffing",
            "NominalDuration":float(self.sub_t_polish*60.0),
            "NominalCost":float(self.sub_c_polish)
        },
        {
            "Name":"Inspection",
            "RequiredCapability":"Buffing",
            "NominalDuration":float(self.sub_t_inspection*60.0),
            "NominalCost":float(self.sub_c_inspection )             
        }])
        if guess:
            if not os.path.exists(os.path.join(self.directory_path,"ProbePlans")):
                os.makedirs(os.path.join(self.directory_path,"ProbePlans"))
            with open(os.path.join(self.directory_path,"ProbePlans",f"ProbePlans_{guess}.json"),"w") as json_file:
                json.dump(self.p_plans[0],json_file, indent=4)

            #Supply Chain Scheduler Call
            request = json.load(open(os.path.join(self.directory_path,f"ProbePlans/ProbePlans_{guess}.json")))
            r = requests.post('http://localhost:9090/generate-bid', data=json.dumps(request))
            print(f"Status Code: {r.status_code}, Response: {r.json()}")
            data = r.json()["data"]
            if data is not None:
                bids_path = os.path.join(self.directory_path,"ProbePlans/Bids")
                if not os.path.exists(bids_path):
                    os.mkdir(bids_path)
                for i in range(len(data)):
                    fname = os.path.join(bids_path,f"{guess}_Bid_{i}.json")
                    with open(fname,"w") as json_file:
                            json.dump(data[i],json_file, indent=4)
                    self.probe_dict['Supplier'].append(data[i]['suppliers'])
                    self.probe_dict['Lead Time (min)'].append(round((data[i]["leadTime"])/60,2))
                    self.probe_dict['Cost ($)'].append(data[i]["cost"])
                    if i>0:
                        self.probe_dict['Manufacturing Method'].append(self.probe_dict['Manufacturing Method'][-1])
                        self.probe_dict['Material'].append(self.probe_dict['Material'][-1])
                        self.probe_dict['Probe'].append(self.probe_dict['Probe'][-1])
                        self.probe_dict['Mass (g)'].append(self.probe_dict['Mass (g)'][-1])
                        self.probe_dict['Compliance (Nm)'].append(self.probe_dict['Compliance (Nm)'][-1])
                        self.probe_dict['Nominal Time (min)'].append(self.probe_dict['Nominal Time (min)'][-1])
                        self.probe_dict['Nominal Cost ($)'].append(self.probe_dict['Nominal Cost ($)'][-1])
            else:
                print('Probe Scheduler returned None bid')
                self.probe_dict['Manufacturing Method'] = self.probe_dict['Manufacturing Method'][:-1]
                self.probe_dict['Material'] =  self.probe_dict['Material'][:-1]
                self.probe_dict['Probe'] = self.probe_dict['Probe'][:-1]
                self.probe_dict['Mass (g)'] = self.probe_dict['Mass (g)'][:-1]
                self.probe_dict['Compliance (Nm)'] = self.probe_dict['Compliance (Nm)'][:-1]
                self.probe_dict['Nominal Time (min)'] = self.probe_dict['Nominal Time (min)'][:-1]
                self.probe_dict['Nominal Cost ($)'] = self.probe_dict['Nominal Cost ($)'][:-1]
        else:
            with open(os.path.join(self.directory_path,"PartProcessPlans.json"),"w") as json_file:
                json.dump(self.p_plans[0],json_file, indent=4)
            t_total, c_total = self.time_cost_eval()
            self.data_dict['Nominal Time (min)'].append(t_total.numpy())
            self.data_dict['Nominal Cost ($)'].append(c_total.numpy())
    def save_a_process_plan(self,guess=None):
        if guess:
            self.probe_dict['Manufacturing Method'].append('Additive '+str(self.orientation[:]))
            self.probe_dict['Material'].append(self.mat_name)
            self.probe_dict['Probe'].append(guess)
            xPhys = np.ones([self.nely,self.nelx,self.nelz],dtype = np.float32) * guess
            self.mass_guess = float(np.sum(xPhys) *self.lele**3 * self.m_density* 1000.0)
            self.comp_guess = float((self.fea.compliance_cp(tf.convert_to_tensor(tf.reshape(xPhys,[self.nely,self.nelx,self.nelz]),dtype=tf.float32))*self.fea.E0).numpy())
            self.time_guess,self.cost_guess = self.time_cost_eval(guess_vf=guess)
            self.probe_dict['Mass (g)'].append(self.mass_guess)
            self.probe_dict['Compliance (Nm)'].append(self.comp_guess)
            self.probe_dict['Nominal Time (min)'].append(float(self.time_guess.numpy()))
            self.probe_dict['Nominal Cost ($)'].append(float(self.cost_guess.numpy()))
            c = self.comp_guess

        else:
            xPhys = self.rbnn(self.dlX).numpy().reshape([self.nely,self.nelx,self.nelz])
            cutoff = self.cutoff_calc(xPhys)
            xPhysc = tf.convert_to_tensor(1.0*(xPhys>cutoff),dtype = tf.float32)
            self.fea.penal = 3.0
            c = float((self.fea.compliance_cp(tf.cast(tf.reshape(xPhysc,[self.nely,self.nelx,self.nelz]),dtype=tf.float32))*self.fea.E0).numpy())

        xPhysf = (xPhys>0.4).astype(float)*0.5
        if guess:
            sv_total = tf.cast(xPhys*0.1,dtype=tf.float32)
        else:
            sv_total = self.sv_loss()
        sv_density = 0.3
        sv_mass = tf.reduce_sum(sv_total)*self.lele**3 * sv_density * self.m_density 
        part_mass = tf.convert_to_tensor(np.sum(xPhys) *self.lele**3 * self.m_density, dtype = tf.float32)
        total_mass = sv_mass + part_mass
        print_time = total_mass / self.am_print_rate 
        # svp = (sv_total.numpy()>0.2).astype(float)
        svp = sv_total.numpy().astype(float)
        if not guess:
            plot3d(xPhysf+svp,save = True)#, title = "support material (red)")     
            plt.savefig(os.path.join(self.directory_path, "support.png"), transparent=True)

        self.p_plans[0]["ProcessPlans"].extend([{
                "Name":"SLM-Additive-1",
                "Part": "Bracket",
                "Link": self.directory_path,
                "Material": self.mat_name + 'powder',
                "ManufacturingMethod":"SLM-Additive",
                "BulkGrams": float(total_mass* 1000.0),
                "NetGrams":float(part_mass* 1000.0),
                "ScrapGrams": float(sv_mass * 1000.0),
                "Width":float(self.nelx*self.lele*1000.0),
                "Height":float(self.nely*self.lele*1000.0),
                "Depth":float(self.nelz*self.lele*1000.0),
                "Compliance":c,
                "Volume":float(np.sum(xPhys) *self.lele**3 * 1e9),
                "Task_Sequence":[{
                    "Name":"Additive-machine-setup",
                    "RequiredCapability":"SLM-Metal-Printing",
                    "NominalDuration":float(self.add_t_setup*60.0),
                    "NominalCost":float(self.add_c_setup)
                    },
                    {
                    "Name":"SLM-printing",
                    "RequiredCapability":"SLM-Metal-Printing",
                    "NominalDuration":float(print_time*60.0),
                    "NominalCost":float(print_time*self.add_m_cost_per_min),
                    },
                    {
                    "Name":"Support-removal",
                    "RequiredCapability":"Buffing",
                    "NominalDuration":float(self.add_t_support*60.0),
                    "NominalCost":float(self.add_c_support)
                    },
                    {
                    "Name":"Inspection",
                    "RequiredCapability":"Buffing",
                    "NominalDuration":float(self.add_t_inspection*60.0),
                    "NominalCost":float(self.add_c_inspection)                                       
                    }]
                }])

        if guess:
            if not os.path.exists(os.path.join(self.directory_path,"ProbePlans")):
                os.makedirs(os.path.join(self.directory_path,"ProbePlans"))
            with open(os.path.join(self.directory_path,"ProbePlans",f"ProbePlans_{guess}.json"),"w") as json_file:
                json.dump(self.p_plans[0],json_file, indent=4)

            #Supply Chain Scheduler Call
            request = json.load(open(os.path.join(self.directory_path,f"ProbePlans/ProbePlans_{guess}.json")))
            r = requests.post('http://localhost:9090/generate-bid', data=json.dumps(request))
            print(f"Status Code: {r.status_code}, Response: {r.json()}")
            data = r.json()["data"]
            if data is not None:
                bids_path = os.path.join(self.directory_path,"ProbePlans/Bids")
                if not os.path.exists(bids_path):
                    os.mkdir(bids_path)
                for i in range(len(data)):
                    fname = os.path.join(bids_path,f"{guess}_Bid_{i}.json")
                    with open(fname,"w") as json_file:
                            json.dump(data[i],json_file, indent=4)
                    self.probe_dict['Supplier'].append(data[i]['suppliers'])
                    self.probe_dict['Lead Time (min)'].append(round((data[i]["leadTime"])/60,2))
                    self.probe_dict['Cost ($)'].append(data[i]["cost"])
                    if i>0:
                        self.probe_dict['Manufacturing Method'].append(self.probe_dict['Manufacturing Method'][-1])
                        self.probe_dict['Material'].append(self.probe_dict['Material'][-1])
                        self.probe_dict['Probe'].append(self.probe_dict['Probe'][-1])
                        self.probe_dict['Mass (g)'].append(self.probe_dict['Mass (g)'][-1])
                        self.probe_dict['Compliance (Nm)'].append(self.probe_dict['Compliance (Nm)'][-1])
                        self.probe_dict['Nominal Time (min)'].append(self.probe_dict['Nominal Time (min)'][-1])
                        self.probe_dict['Nominal Cost ($)'].append(self.probe_dict['Nominal Cost ($)'][-1])
            else:
                print('Probe Scheduler returned None bid')
                self.probe_dict['Manufacturing Method'] = self.probe_dict['Manufacturing Method'][:-1]
                self.probe_dict['Material'] =  self.probe_dict['Material'][:-1]
                self.probe_dict['Probe'] = self.probe_dict['Probe'][:-1]
                self.probe_dict['Mass (g)'] = self.probe_dict['Mass (g)'][:-1]
                self.probe_dict['Compliance (Nm)'] = self.probe_dict['Compliance (Nm)'][:-1]
                self.probe_dict['Nominal Time (min)'] = self.probe_dict['Nominal Time (min)'][:-1]
                self.probe_dict['Nominal Cost ($)'] = self.probe_dict['Nominal Cost ($)'][:-1]

        else:
            with open(os.path.join(self.directory_path,"PartProcessPlans.json"),"w") as json_file:
                json.dump(self.p_plans[0],json_file, indent=4)
            t_total, c_total = self.time_cost_eval()
            self.data_dict['Nominal Time (min)'].append(t_total.numpy())
            self.data_dict['Nominal Cost ($)'].append(c_total.numpy())


    def save_c_process_plan(self,guess=None):
        if guess:
            self.probe_dict['Manufacturing Method'].append('EDM '+str(self.orientation[:]))
            self.probe_dict['Material'].append(self.mat_name)
            self.probe_dict['Probe'].append(guess)
            xPhys = np.ones([self.nely,self.nelx,self.nelz],dtype = np.float32) * guess
            self.mass_guess = float(np.sum(xPhys) *self.lele**3 * self.m_density* 1000.0)
            self.comp_guess = float((self.fea.compliance_cp(tf.convert_to_tensor(tf.reshape(xPhys,[self.nely,self.nelx,self.nelz]),dtype=tf.float32))*self.fea.E0).numpy())
            self.time_guess,self.cost_guess = self.time_cost_eval(guess_vf=guess)
            self.probe_dict['Mass (g)'].append(self.mass_guess)
            self.probe_dict['Compliance (Nm)'].append(self.comp_guess)
            self.probe_dict['Nominal Time (min)'].append(float(self.time_guess))
            self.probe_dict['Nominal Cost ($)'].append(float(self.cost_guess))
            c = self.comp_guess
        else:
            xPhys = self.rbnn(self.dlX).numpy().reshape([self.nely,self.nelx,self.nelz])
            cutoff = self.cutoff_calc(xPhys)
            xPhysc = tf.convert_to_tensor(1.0*(xPhys>cutoff),dtype = tf.float32)
            self.fea.penal = 3.0
            c = float((self.fea.compliance_cp(tf.cast(tf.reshape(xPhysc,[self.nely,self.nelx,self.nelz]),dtype=tf.float32))*self.fea.E0).numpy())
        xPhysf = (xPhys>0.4).astype(float)*0.5

        if guess:
            cut_area = self.nele *self.lele**2 * (1.0-guess)
            t_cut = cut_area / self.q_cut
        else:
            img = np.mean(xPhys,axis = np.argmax(self.c_dir))
            nv = np.sum(img)
            N1 = img.shape[0]
            N2 = img.shape[1]
            i = img[0:N1,0:N2-1]
            j = img[0:N1,1:N2]
            sum_n1 = np.multiply(i,j)
            n1 = nv - np.sum(sum_n1)
            i = img[0:N1-1,0:N2]
            j = img[1:N1,0:N2]
            sum_n2 = np.multiply(i,j)
            n2 = nv - np.sum(sum_n2)
            i = img[0:N1-1,0:N2-1]
            j = img[1:N1, 1:N2]
            sum_n3 = np.multiply(i,j)
            n3 = nv - np.sum(sum_n3)
            i = img[0:N1-1, 1:N2]
            j = img[1:N1, 0:N2-1]
            sum_n4 = np.multiply(i,j)
            n4 = nv - np.sum(sum_n4)
            perim = m.pi * 0.25* (n1 + n2 + n3/2**0.5 + n4/2**0.5)
            cut_area = perim * [self.nely,self.nelx,self.nelz][np.argmax(self.c_dir)] * self.lele **2
            #cut_area = self.edm_area() * self.lele **2
            t_cut = cut_area / self.q_cut


            plot3d((0.5 - xPhysf)*0.6+ xPhysf ,save = True)
            plt.savefig(os.path.join(self.directory_path, "cutting_removal.png"), transparent=True)

        self.p_plans[0]["ProcessPlans"].extend([{
            "Name":"2-Axis-Subtractive-1",
            "Part": "Bracket",
            "Link": self.directory_path,
            "Material": self.mat_name,
            "ManufacturingMethod":"2-Axis-Subtractive",
            "BulkGrams": float(self.nele* self.lele**3 * self.m_density * 1000.0),
            "NetGrams":float(np.sum(xPhys) *self.lele**3 * self.m_density* 1000.0),
            "ScrapGrams": float((self.nele - np.sum(xPhys)) * self.lele**3 * self.m_density * 1000.0),
            "Width":float(self.nelx*self.lele*1000.0),
            "Height":float(self.nely*self.lele*1000.0),
            "Depth":float(self.nelz*self.lele*1000.0),
            "Compliance":c,
            "Volume":float(np.sum(xPhys) *self.lele**3 * 1e9),
            "Task_Sequence":[{
                "Name":"EDM-machine-setup",
                "RequiredCapability":"EDM-Machining",
                "NominalDuration":float(self.edm_t_setup*60.0),
                "NominalCost":float(self.edm_c_setup)
                },
                {
                "Name":"2Axis-machining-1",
                "RequiredCapability":"EDM-Machining",
                "NominalDuration":float(t_cut*60.0),
                "NominalCost":float(t_cut*self.edm_m_cost_per_min),
                },
                {
                "Name":"Polishing",
                "RequiredCapability":"Buffing",
                "NominalDuration":float(self.edm_t_polish*60.0),
                "NominalCost":float(self.edm_c_polish)
                },
                {
                "Name":"Inspection",
                "RequiredCapability":"Buffing",
                "NominalDuration":float(self.edm_t_inspection*60.0),
                "NominalCost":float(self.edm_c_inspection)                                       
                }]
            }])
        if guess:
            if not os.path.exists(os.path.join(self.directory_path,"ProbePlans")):
                os.makedirs(os.path.join(self.directory_path,"ProbePlans"))
            with open(os.path.join(self.directory_path,"ProbePlans",f"ProbePlans_{guess}.json"),"w") as json_file:
                json.dump(self.p_plans[0],json_file, indent=4)

            #Supply Chain Scheduler Call
            request = json.load(open(os.path.join(self.directory_path,f"ProbePlans/ProbePlans_{guess}.json")))
            r = requests.post('http://localhost:9090/generate-bid', data=json.dumps(request))
            print(f"Status Code: {r.status_code}, Response: {r.json()}")
            data = r.json()["data"]
            if data is not None:
                bids_path = os.path.join(self.directory_path,"ProbePlans/Bids")
                if not os.path.exists(bids_path):
                    os.mkdir(bids_path)
                for i in range(len(data)):
                    fname = os.path.join(bids_path,f"{guess}_Bid_{i}.json")
                    with open(fname,"w") as json_file:
                            json.dump(data[i],json_file, indent=4)
                    self.probe_dict['Supplier'].append(data[i]['suppliers'])
                    self.probe_dict['Lead Time (min)'].append(round((data[i]["leadTime"])/60,2))
                    self.probe_dict['Cost ($)'].append(data[i]["cost"])
                    if i>0:
                        self.probe_dict['Manufacturing Method'].append(self.probe_dict['Manufacturing Method'][-1])
                        self.probe_dict['Material'].append(self.probe_dict['Material'][-1])
                        self.probe_dict['Probe'].append(self.probe_dict['Probe'][-1])
                        self.probe_dict['Mass (g)'].append(self.probe_dict['Mass (g)'][-1])
                        self.probe_dict['Compliance (Nm)'].append(self.probe_dict['Compliance (Nm)'][-1])
                        self.probe_dict['Nominal Time (min)'].append(self.probe_dict['Nominal Time (min)'][-1])
                        self.probe_dict['Nominal Cost ($)'].append(self.probe_dict['Nominal Cost ($)'][-1])
            else:
                print('Probe Scheduler returned None bid')
                self.probe_dict['Manufacturing Method'] = self.probe_dict['Manufacturing Method'][:-1]
                self.probe_dict['Material'] =  self.probe_dict['Material'][:-1]
                self.probe_dict['Probe'] = self.probe_dict['Probe'][:-1]
                self.probe_dict['Mass (g)'] = self.probe_dict['Mass (g)'][:-1]
                self.probe_dict['Compliance (Nm)'] = self.probe_dict['Compliance (Nm)'][:-1]
                self.probe_dict['Nominal Time (min)'] = self.probe_dict['Nominal Time (min)'][:-1]
                self.probe_dict['Nominal Cost ($)'] = self.probe_dict['Nominal Cost ($)'][:-1]
        else:
            with open(os.path.join(self.directory_path,"PartProcessPlans.json"),"w") as json_file:
                json.dump(self.p_plans[0],json_file, indent=4)

            mat_cost = self.nele *self.lele**3 * self.m_density * self.c_mat
            t_total = self.edm_t_setup + t_cut + self.edm_t_inspection +self.edm_t_polish
            c_total = self.edm_c_setup + t_cut*self.edm_m_cost_per_min + self.edm_c_polish + self.edm_c_inspection + mat_cost
            self.data_dict['Nominal Time (min)'].append(t_total)
            self.data_dict['Nominal Cost ($)'].append(c_total)

    def save_result(self):

        np.save(os.path.join(self.directory_path, 'weights.npy'),self.weights1.numpy())
        np.save(os.path.join(self.directory_path, 'kernel.npy'),self.kernel.numpy())
        np.save(os.path.join(self.directory_path, 'bias1.npy'),self.bias1.numpy())
        np.save(os.path.join(self.directory_path, 'bias2.npy'),self.bias2.numpy())
        # np.save(os.path.join(self.directory_path, 'loc_c.npy'), np.array(self.log_compliance))

        xPhys = self.rbnn(self.dlX).numpy().reshape([self.nely,self.nelx,self.nelz])
        # xPhysSS = self.rbnn(self.dlXSS).numpy().reshape([self.nely*2,self.nelx*2,self.nelz*2])
        try:
            np.save(os.path.join(self.directory_path, 'xPhys.npy'), xPhys)
            plot3d(xPhys, save=True)   
            plt.savefig(os.path.join(self.directory_path, 'xPhys.png'), transparent=True)
            cutoff = self.cutoff_calc(xPhys)
            plot_iso(self, xPhys, cutoff, save=True)   
            plt.savefig(os.path.join(self.directory_path, 'xPhys_iso.png'), transparent=True)
            xPhysc = tf.convert_to_tensor(1.0*(xPhys>cutoff),dtype = tf.float32)
        except:
            print('Failed Rendering')

        # print('penalty: ',self.fea.penal)
        comp = float((self.fea.compliance_cp(tf.cast(tf.reshape(xPhysc,[self.nely,self.nelx,self.nelz]),dtype=tf.float32))*self.fea.E0).numpy())
        max_d = float(self.fea.max_disp(tf.reshape(xPhysc,[self.nely,self.nelx,self.nelz]))*1000)
        mass = tf.reduce_sum(xPhys) *self.lele**3 * self.m_density
        self.data_dict['Mass (g)'].append(mass.numpy()*1000)
        self.data_dict['Compliance (Nm)'].append(comp)
        self.data_dict['Maximum Displacement (mm)'].append(max_d)
        print('max d: ',max_d)
        print('comp: ',comp)
        

        self.p_plans = json.load(open(self.request_header_json))
        #save process plans 
        if self.unrestricted:
            process_plan = np.array([('unrestricted', 0,0)], dtype = [('name', 'U11'), ('time', 'f4'), ('cost', 'f4')])
            mat_cost = np.sum(xPhys) *self.lele**3 * self.m_density * self.c_mat

        elif sum(self.c_dir)>0: #edm cutting
            self.save_c_process_plan()


        elif sum(self.m_dir) >0: #subtractive
            self.save_m_process_plan()


        elif sum(self.a_dir)>0: #additive
            self.save_a_process_plan()

  
    def gen_probe(self):
        vf_list = [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.01,0.005]
        if sum(self.c_dir)>0: #edm cutting
            for vf in vf_list:
                self.p_plans = json.load(open(self.request_header_json))
                self.save_c_process_plan(guess=vf)    

        elif sum(self.m_dir) >0: #subtractive
            for vf in vf_list:
                self.p_plans = json.load(open(self.request_header_json))
                self.save_m_process_plan(guess=vf)    

        elif sum(self.a_dir)>0: #additive
            for vf in vf_list:
                self.p_plans = json.load(open(self.request_header_json))
                self.save_a_process_plan(guess=vf)        


        #MinMax
        if 'Bids' in os.listdir(os.path.join(self.directory_path,"ProbePlans")):
            bids_path = os.path.join(self.directory_path,"ProbePlans/Bids")
            json_data = {"mass_min":min(self.probe_dict['Mass (g)']), "mass_max":max(self.probe_dict['Mass (g)']),"comp_min":min(self.probe_dict['Compliance (Nm)']),"comp_max":max(self.probe_dict['Compliance (Nm)']),
                            "time_min":min(self.probe_dict['Lead Time (min)']),"time_max":max(self.probe_dict['Lead Time (min)']),"cost_min":min(self.probe_dict['Cost ($)']),"cost_max":max(self.probe_dict['Cost ($)'])}
            with open(bids_path+'/Bids_Min_Max.json',"w") as json_file:
                json.dump(json_data,json_file, indent=4)



def mass_time_cost_vf_evaluator(opt1, mmm, design_probe_df, rh, mass_con, time_con, cost_con):
    curr_nele_m = mass_con/(opt1.lele**3 * opt1.m_density)
    mass_vf = curr_nele_m/opt1.nele
    if mass_vf>=1.0:
        mass_vf = 0.99999

    df_list = []
    # time_vf_list = []
    # cost_vf_list = []
    num_suppliers = len(rh[0]['SupplierNames'])
    vf_arr = np.zeros((2,num_suppliers))
    for i in range(num_suppliers):
        df = design_probe_df[design_probe_df['Supplier'] == '{}'.format(rh[0]['SupplierNames'][i])]
        df_list.append(df)
        if len(df) !=0:
            #Lead Time fitting
            x = df["Lead Time (min)"].tolist()
            y = df["Probe"].tolist()
            xs_eval_list = [abs(x[i] - time_con) for i in range(len(x))]
            xs_sorted = sorted(xs_eval_list)
            ind1 = xs_eval_list.index(xs_sorted[0])
            if len(x)>1:
                ind2 = xs_eval_list.index(xs_sorted[1])
            else:
                ind2 = ind1
            if ind2<ind1:
                ind1,ind2 = ind2,ind1
            x = np.array([x[ind1],x[ind2]]).reshape((-1,1))
            y = np.array([y[ind1],y[ind2]]).reshape((-1,1))
            model = LinearRegression()
            model.fit(x, y)
            # time_vf_list.append(float(model.predict((np.array(time_con).reshape(-1,1))).reshape(1)))
            vf_arr[0,i]=(float(model.predict((np.array(time_con).reshape(-1,1))).reshape(1)))
            #Cost fitting
            x = np.array(df["Cost ($)"]).tolist()
            y = np.array(df["Probe"]).tolist()
            xs_eval_list = [abs(x[i] - cost_con) for i in range(len(x))]
            xs_sorted = sorted(xs_eval_list)
            ind1 = xs_eval_list.index(xs_sorted[0])
            if len(x)>1:
                ind2 = xs_eval_list.index(xs_sorted[1])
            else:
                ind2 = ind1
            if ind2<ind1:
                ind1,ind2 = ind2,ind1
            x = np.array([x[ind1],x[ind2]]).reshape((-1,1))
            y = np.array([y[ind1],y[ind2]]).reshape((-1,1))
            model = LinearRegression()
            model.fit(x, y)
            # cost_vf_list.append(float(model.predict((np.array(cost_con).reshape(-1,1))).reshape(1)))
            vf_arr[1,i]= (float(model.predict((np.array(cost_con).reshape(-1,1))).reshape(1)))
    if mmm["ManufacturingMethod"] == "Additive":
        # time_vf = max(time_vf_list)
        # cost_vf = max(cost_vf_list)
        best_supplier_num = np.argmax(np.min(vf_arr,axis = 0))
        time_vf = float(vf_arr[0,best_supplier_num])
        cost_vf = float(vf_arr[1,best_supplier_num])
    else:
        # time_vf = min(time_vf_list)
        # cost_vf = min(cost_vf_list)
        best_supplier_num = np.argmin(np.max(vf_arr,axis = 0))
        time_vf = float(vf_arr[0,best_supplier_num])
        cost_vf = float(vf_arr[1,best_supplier_num])
    return mass_vf,time_vf,cost_vf

def initialize_comp_and_vfs_in_objective(opt1,mmm, mass_con, mass_vf, cost_vf, time_vf):
    opt1.mass_con = mass_con
    print('cost_vf')
    _,opt1.cost_con = opt1.time_cost_eval(cost_vf)
    opt1.time_con,_ = opt1.time_cost_eval(time_vf)
    print('opt1.cost_con',opt1.cost_con)

    if mmm["ManufacturingMethod"] == "Additive":
        opt1.init_vf = min(mass_vf,time_vf,cost_vf)
        if opt1.init_vf<1.0:
            opt1.offset = tf.math.log(opt1.init_vf/(1-opt1.init_vf))
        else:
            opt1.offset = tf.math.log(0.99999/(1-0.99999))
    else:
        opt1.init_vf = mass_vf
        if opt1.init_vf<1.0:
            opt1.offset = tf.math.log(mass_vf/(1-mass_vf))
        else:
            opt1.offset = tf.math.log(0.99999/(1-0.99999))

    if opt1.init_vf<0.2:
        xPhys_c0 = tf.ones([opt1.nely, opt1.nelx, opt1.nelz])*0.2
        opt1.c_0 = opt1.fea.compliance_cp(xPhys_c0)
    elif opt1.init_vf>0.2 and opt1.init_vf<1.0:
        xPhys_c0 = tf.ones([opt1.nely, opt1.nelx, opt1.nelz])*opt1.init_vf
        opt1.c_0 = opt1.fea.compliance_cp(xPhys_c0)
    else:
        xPhys_c0 = tf.ones([opt1.nely, opt1.nelx, opt1.nelz])
        opt1.c_0 = opt1.fea.compliance_cp(xPhys_c0)
        

def run_opt(request_header_json = 'request_header.json', mmm_json = 'mmm.json',bc_json = 'bc.json',mat_lib_json = 'mat_lib.json',
            machine_json = 'machine.json', constraints_json = 'constraints.json', probe=False, test = False):

    directory_path = os.path.dirname(mmm_json)
    mmm = json.load(open(mmm_json))
    bc = json.load(open(bc_json))

    mat_lib = json.load(open(mat_lib_json))
    machine =  json.load(open(machine_json))

    constraints = json.load(open(constraints_json))
    mass_con = int(constraints["Mass_constraint"][0])
    cost_con = int(constraints["Cost_constraint"][0])
    time_con = int(constraints["Time_constraint"][0])


    lele = bc["lele"]
    nelx = bc["nelx"]
    nely = bc["nely"]
    nelz = bc["nelz"]
    nelm = max(nelx,nely,nelz)
    volfrac = 0.3

    E0 = mat_lib[mmm["Material"]]["E0"]
    Emin = mat_lib[mmm["Material"]]["Emin"]
    nu = mat_lib[mmm["Material"]]["nu"]

    #define boundary condition
    il = np.array(bc["il"],dtype = float)
    jl = np.array(bc["jl"],dtype = float)
    kl = np.array(bc["kl"],dtype = float)
    il_F = np.array(bc["il_F"],dtype = float)
    jl_F = np.array(bc["jl_F"],dtype = float)
    kl_F = np.array(bc["kl_F"],dtype = float)

    iif = np.array(bc["iif"],dtype = float).reshape([-1,1,1])
    jf = np.array(bc["jf"],dtype = float).reshape([-1,1,1])
    kf = np.array(bc["kf"],dtype = float).reshape([-1,1,1])
        

    #initiate and run 
    opt1 = nnopt(nely,nelx,nelz,volfrac,lele,directory_path)
    opt1.debug = False
    opt1.init_mat(E0, Emin, nu,mmm["Material"])
    opt1.c_mat = mat_lib[mmm["Material"]]["mat_cost"]
    opt1.c_mat_powder = mat_lib[mmm["Material"]]["powder_cost"]
    opt1.m_density = mat_lib[mmm["Material"]]["density"]
    opt1.am_print_rate = mat_lib[mmm["Material"]]["am_print_rate"]
    opt1.m_removal_rate = mat_lib[mmm["Material"]]["m_removal_rate"]
    opt1.q_cut = mat_lib[mmm["Material"]]["q_cut"]
    opt1.request_header_json = request_header_json
    opt1.mat_lib_file = mat_lib
    opt1.mmm_file = mmm

    opt1.data_dict = collections.defaultdict(list)
    opt1.data_dict['Mass Constraint (g)'].append(mass_con*1000)
    opt1.data_dict['Cost Constraint ($)'].append(cost_con)
    opt1.data_dict['Time Constraint (min)'].append(time_con)
    opt1.probe_dict = collections.defaultdict(list)


    if mmm["ManufacturingMethod"] == "Subtractive" :
        m_names = np.array(['y+', 'y-', 'x+', 'x-', 'z+', 'z-'])
        orientation = mmm["Orientation"]
        opt1.orientation = orientation
        orientation_bool = np.array([0,0,0,0,0,0])

        orientation_bool[np.where(np.isin(m_names,orientation))] = 1
        opt1.init_3axis(orientation_bool.tolist())
        
        opt1.sub_t_setup = machine["Subtractive"]["t_setup"]
        opt1.sub_c_setup = machine["Subtractive"]["c_setup"]
        opt1.sub_t_opt_setup = machine["Subtractive"]["t_opt_setup"]
        opt1.sub_c_opt_setup = machine["Subtractive"]["c_opt_setup"]
        opt1.sub_m_cost_per_min = machine["Subtractive"]["m_cost_per_min"]
        opt1.sub_t_polish = machine["Subtractive"]["t_polish"]
        opt1.sub_c_polish = machine["Subtractive"]["c_polish"]
        opt1.sub_t_inspection = machine["Subtractive"]["t_inspection"]
        opt1.sub_c_inspection = machine["Subtractive"]["c_inspection"]

        opt1.data_dict['Manufacturing Method'].append('Subtractive '+orientation[0])

    elif mmm["ManufacturingMethod"] == "Additive" :
        m_names = np.array(['y+', 'y-', 'x+', 'x-', 'z+', 'z-'])
        orientation = mmm["Orientation"]
        opt1.orientation = orientation
        orientation_bool = np.array([0,0,0,0,0,0])

        orientation_bool[np.where(np.isin(m_names,orientation))] = 1
        opt1.init_m_additive(orientation_bool.tolist())

        opt1.add_t_setup = machine["Additive"]["t_setup"]
        opt1.add_c_setup = machine["Additive"]["c_setup"]
        opt1.add_m_cost_per_min = machine["Additive"]["m_cost_per_min"]
        opt1.add_t_support = machine["Additive"]["t_support"]
        opt1.add_c_support = machine["Additive"]["c_support"]
        opt1.add_t_inspection = machine["Additive"]["t_inspection"]
        opt1.add_c_inspection = machine["Additive"]["c_inspection"]

        opt1.data_dict['Manufacturing Method'].append('Additive '+orientation[0])

    elif mmm["ManufacturingMethod"] == "EDM" :
        m_names = np.array(['y', 'x', 'z'])
        orientation = mmm["Orientation"]
        opt1.orientation = orientation
        orientation_bool = np.array([0,0,0])

        orientation_bool[np.where(np.isin(m_names,orientation))] = 1
        opt1.init_2axis(orientation_bool.tolist())

        opt1.edm_t_setup = machine["EDM"]["t_setup"]
        opt1.edm_c_setup = machine["EDM"]["c_setup"]
        opt1.edm_m_cost_per_min = machine["EDM"]["m_cost_per_min"]
        opt1.edm_t_polish = machine["EDM"]["t_polish"]
        opt1.edm_c_polish = machine["EDM"]["c_polish"]
        opt1.edm_t_inspection = machine["EDM"]["t_inspection"]
        opt1.edm_c_inspection = machine["EDM"]["c_inspection"]

        opt1.data_dict['Manufacturing Method'].append('EDM '+orientation[0])


    #Initialize FEA solver
    opt1.fea.E0 = mat_lib[mmm["Material"]]["E0"]
    opt1.fea.KE =  lk_H8_np(nu)*opt1.lele
    opt1.init_bc(il, jl, kl, il_F, jl_F,  kl_F, iif, jf, kf)


    # if 'probe_results.csv' not in os.listdir(directory_path):
    if probe:
        xPhys_c0 = tf.ones([opt1.nely, opt1.nelx, opt1.nelz])
        opt1.c_0 = opt1.fea.compliance_cp(xPhys_c0)
        opt1.gen_probe()
        design_probe_df = pd.DataFrame(data = opt1.probe_dict)
        design_probe_df.to_csv(os.path.join(directory_path, 'probe_results.csv'),index = False)
    else:
        # try:
        design_probe_df = pd.read_csv(os.path.join(directory_path, 'probe_results.csv'))
        # except FileNotFoundError:
        #         print("Probes not generated")

    feasible = True
    #Calculate constraint dependent volume fractions
    if len(design_probe_df) == 0:
        print('len(design_probe_df)')
        feasible = False
    elif  time_con < min(design_probe_df['Lead Time (min)']) or cost_con < min(design_probe_df['Cost ($)']):
        print('con supplier')
        feasible = False
    if feasible:
        rh = json.load(open(opt1.request_header_json))
        mass_vf,time_vf,cost_vf = mass_time_cost_vf_evaluator(opt1,mmm,design_probe_df,rh,mass_con,time_con,cost_con)
        print('mass_vf,time_vf,cost_vf',mass_vf,time_vf,cost_vf)
        initialize_comp_and_vfs_in_objective(opt1,mmm, mass_con, mass_vf, cost_vf, time_vf)

    if not test:
        #Infeasible Suppliers
        if not feasible:
            opt1.data_dict['Mass (g)'].append("infeasible")
            opt1.data_dict['Compliance (Nm)'].append("infeasible")
            opt1.data_dict['Maximum Displacement (mm)'].append("infeasible")
            opt1.data_dict['Nominal Time (min)'].append("infeasible")
            opt1.data_dict['Nominal Cost ($)'].append("infeasible")
            opt1.data_dict['Supplier'].append("infeasible")
            opt1.data_dict['Lead Time (min)'].append("infeasible")
            opt1.data_dict['Cost ($)'].append("infeasible")
            opt1.data_dict['Material'].append(mmm["Material"])
            design_ai_data = pd.DataFrame(data = opt1.data_dict)
            design_ai_data.to_csv(os.path.join(directory_path, 'final_results.csv'),index = False)
            mmm["Valid"] = False
            mmm["Reason"] = "Suppliers Unavailable for Current Configuration"
            with open(os.path.join(opt1.directory_path,"mmm.json"),"w") as json_file:
                json.dump(mmm,json_file, indent=4)
            

        #Infeasible constraints
        elif (((mmm["ManufacturingMethod"] == "Subtractive" or mmm["ManufacturingMethod"] == "EDM") and (mass_vf<cost_vf or mass_vf<time_vf or mass_vf<0.02))
                or ((mmm["ManufacturingMethod"] == "Additive") and (min(mass_vf,cost_vf,time_vf)<0.02))):
        # elif (mmm["ManufacturingMethod"] == "Subtractive") and (mass_vf<cost_vf or mass_vf<time_vf):
            print('mass_vf: ',mass_vf, 'cost_vf: ',cost_vf,'time_vf: ',time_vf)
            # print(True)
            opt1.data_dict['Mass (g)'].append("infeasible")
            opt1.data_dict['Compliance (Nm)'].append("infeasible")
            opt1.data_dict['Maximum Displacement (mm)'].append("infeasible")
            opt1.data_dict['Nominal Time (min)'].append("infeasible")
            opt1.data_dict['Nominal Cost ($)'].append("infeasible")
            opt1.data_dict['Supplier'].append("infeasible")
            opt1.data_dict['Lead Time (min)'].append("infeasible")
            opt1.data_dict['Cost ($)'].append("infeasible")
            opt1.data_dict['Material'].append(mmm["Material"])
            design_ai_data = pd.DataFrame(data = opt1.data_dict)
            design_ai_data.to_csv(os.path.join(directory_path, 'final_results.csv'),index = False)
            mmm["Valid"] = False
            mmm["Reason"] = "Infeasible Constraints"
            with open(os.path.join(opt1.directory_path,"mmm.json"),"w") as json_file:
                json.dump(mmm,json_file, indent=4)
            feasible = False

        elif not probe:
            print('mass_vf: ',mass_vf, 'cost_vf: ',cost_vf,'time_vf: ',time_vf)
            #If constraints are too lenient, block of boundary condition envelope is optimal in terms of compliance 
            if (mass_vf>=0.99999 and time_vf>=0.99999 and cost_vf>=0.99999) or (mass_vf>=0.99999 and time_vf<=0.01 and cost_vf<=0.01):
                opt1.offset = tf.math.log(0.99999/(1-0.99999))
                opt1.fit(1)
                opt1.display_result_summary()
                opt1.save_result()
                opt1.data_dict['Material'].append(mmm["Material"])
            else:
                #Initialize FEA solver for optimization
                max_force_val = max(np.max(il_F),np.max(jl_F),np.max(kl_F))
                il_Fn = il_F/max_force_val
                jl_Fn = jl_F/max_force_val
                kl_Fn = kl_F/max_force_val
                opt1.fea.E0 = 1.0
                opt1.fea.KE =  lk_H8_np(0.33)
                opt1.init_bc(il, jl, kl, il_Fn, jl_Fn,  kl_Fn, iif, jf, kf)

                if opt1.init_vf<0.2:
                    xPhys_c0 = tf.ones([opt1.nely, opt1.nelx, opt1.nelz])*0.2
                    opt1.c_0 = opt1.fea.compliance_cp(xPhys_c0)
                else:
                    xPhys_c0 = tf.ones([opt1.nely, opt1.nelx, opt1.nelz])*opt1.init_vf
                    opt1.c_0 = opt1.fea.compliance_cp(xPhys_c0)

                #Run optimization
                opt1.fit(300)

                #Initialize FEA solver for final results
                il_F = np.array(bc["il_F"],dtype = float)
                jl_F = np.array(bc["jl_F"],dtype = float)
                kl_F = np.array(bc["kl_F"],dtype = float)
                opt1.fea.E0 = mat_lib[mmm["Material"]]["E0"]
                opt1.fea.KE =  lk_H8_np(nu)*opt1.lele
                opt1.init_bc(il, jl, kl, il_F, jl_F,  kl_F, iif, jf, kf)
                opt1.fea.penal = 3.0
                if opt1.init_vf<0.2:
                    xPhys_c0 = tf.ones([opt1.nely, opt1.nelx, opt1.nelz])*0.2
                    opt1.c_0 = opt1.fea.compliance_cp(xPhys_c0)
                else:
                    xPhys_c0 = tf.ones([opt1.nely, opt1.nelx, opt1.nelz])*opt1.init_vf
                    opt1.c_0 = opt1.fea.compliance_cp(xPhys_c0)
                    # print('opt1.c_0',opt1.c_0)
                opt1.display_result_summary()
                opt1.save_result()
                opt1.data_dict['Material'].append(mmm["Material"])
            
            mmm["Valid"] = True
            mmm["Reason"] = "Feasible Configuration"
            with open(os.path.join(opt1.directory_path,"mmm.json"),"w") as json_file:
                json.dump(mmm,json_file, indent=4)
            #Supply Chain Scheduler Call
            request = json.load(open(os.path.join(directory_path,"PartProcessPlans.json")))
            r = requests.post('http://localhost:9090/generate-bid', data=json.dumps(request))
            print(f"Status Code: {r.status_code}, Response: {r.json()}")
            data = r.json()["data"]
            if data is not None:
                bids_path = os.path.join(directory_path,"Bids")
                if not os.path.exists(bids_path):
                    os.mkdir(bids_path)
                for i in range(len(data)):
                    fname = os.path.join(bids_path,f"Bid_{i}.json")
                    with open(fname,"w") as json_file:
                            json.dump(data[i],json_file, indent=4)
                    opt1.data_dict['Supplier'].append(data[i]['suppliers'])
                    opt1.data_dict['Lead Time (min)'].append(round((data[i]["leadTime"])/60,2))
                    opt1.data_dict['Cost ($)'].append(data[i]["cost"])
                    if i>0:
                        opt1.data_dict['Mass Constraint (g)'].append(opt1.data_dict['Mass Constraint (g)'][-1])
                        opt1.data_dict['Cost Constraint ($)'].append(opt1.data_dict['Cost Constraint ($)'][-1])
                        opt1.data_dict['Time Constraint (min)'].append(opt1.data_dict['Time Constraint (min)'][-1])
                        opt1.data_dict['Manufacturing Method'].append(opt1.data_dict['Manufacturing Method'][-1])
                        opt1.data_dict['Material'].append(opt1.data_dict['Material'][-1])
                        opt1.data_dict['Mass (g)'].append(opt1.data_dict['Mass (g)'][-1])
                        opt1.data_dict['Compliance (Nm)'].append(opt1.data_dict['Compliance (Nm)'][-1])
                        opt1.data_dict['Maximum Displacement (mm)'].append(opt1.data_dict['Maximum Displacement (mm)'][-1])
                        opt1.data_dict['Nominal Time (min)'].append(opt1.data_dict['Nominal Time (min)'][-1])
                        opt1.data_dict['Nominal Cost ($)'].append(opt1.data_dict['Nominal Cost ($)'][-1])
            else:
                print('Probe Scheduler returned None bid')
                opt1.data_dict['Supplier'].append('None')
                opt1.data_dict['Lead Time (min)'].append('None')
                opt1.data_dict['Cost ($)'].append('None')
            design_ai_data = pd.DataFrame(data = opt1.data_dict)
            design_ai_data.to_csv(os.path.join(directory_path, 'final_results.csv'),index = False)



    elif test:
        if 'kernel.npy' in os.listdir(directory_path):
            k = np.load(directory_path + '\kernel.npy')
            w = np.load(directory_path + '\weights.npy')
            b1 = np.load(os.path.join(directory_path, 'bias1.npy'))
            b2 = np.load(os.path.join(directory_path, 'bias2.npy'))
            opt1.kernel = tf.convert_to_tensor(k,dtype = tf.float32)
            opt1.weights1 = tf.convert_to_tensor(w,dtype = tf.float32)
            opt1.bias1 = tf.convert_to_tensor(b1,dtype = tf.float32)
            opt1.bias2 = tf.convert_to_tensor(b2,dtype = tf.float32)

            #Initialize FEA solver for final results
            il_F = np.array(bc["il_F"],dtype = float)
            jl_F = np.array(bc["jl_F"],dtype = float)
            kl_F = np.array(bc["kl_F"],dtype = float)
            opt1.fea.E0 = mat_lib[mmm["Material"]]["E0"]
            opt1.fea.KE =  lk_H8_np(nu)*opt1.lele
            opt1.init_bc(il, jl, kl, il_F, jl_F,  kl_F, iif, jf, kf)
            opt1.fea.penal = 3.0
            if opt1.init_vf<0.2:
                xPhys_c0 = tf.ones([opt1.nely, opt1.nelx, opt1.nelz])*0.2
                opt1.c_0 = opt1.fea.compliance_cp(xPhys_c0)
            else:
                xPhys_c0 = tf.ones([opt1.nely, opt1.nelx, opt1.nelz])*opt1.init_vf
                opt1.c_0 = opt1.fea.compliance_cp(xPhys_c0)
                print('opt1.c_0',opt1.c_0)

            xPhys = tf.reshape( opt1.rbnn(opt1.dlX) ,[opt1.nely,opt1.nelx,opt1.nelz] ).numpy()
            opt1.display_result_summary()
            # opt1.save_result()

            print('vf:',tf.reduce_mean(xPhys))
            print('0.5 vf:',tf.reduce_mean(1.0*(xPhys>0.5)))
            print('0.2 mass:',tf.reduce_sum(1.0*(xPhys>0.2)) *opt1.lele**3 * opt1.m_density)

            cutoff = opt1.cutoff_calc(xPhys)
            print('cutoff',cutoff)
            print('cutoff mass: ',tf.reduce_sum(1.0*(xPhys>cutoff)) *opt1.lele**3 * opt1.m_density)
            # plot_iso(opt1,xPhys,cutoff,save = True)
            # plot_iso2(xPhys,cutoff,210,120)
            plot_iso2(xPhys,cutoff,210,-120)
            plot_iso2(xPhys,cutoff,-90,0) 
            plot_iso2(xPhys,cutoff,150,-120) 
            xPhysc = tf.convert_to_tensor(1.0*(xPhys>cutoff),dtype = tf.float32) 
            # plt.savefig(os.path.join(opt1.directory_path, 'xPhys_iso_top.png'), transparent=True)
            t_total, c_total = opt1.time_cost_eval()
            opt1.fea.penal = 3.0
            comp = opt1.fea.compliance_cp(tf.reshape(xPhysc,[opt1.nely,opt1.nelx,opt1.nelz]))*opt1.fea.E0
            max_disp_val = float(opt1.fea.max_disp(tf.reshape(xPhysc,[opt1.nely,opt1.nelx,opt1.nelz]))*1000)
            comp2 = opt1.fea.compliance_cp(tf.reshape(xPhys,[opt1.nely,opt1.nelx,opt1.nelz]))*opt1.fea.E0
            mass = tf.reduce_sum(xPhys) *opt1.lele**3 * opt1.m_density

            print("## Result Summary")
            print("VF: ",tf.reduce_mean(xPhys))
            print("Mass: {:.6f} [kg]".format(mass))
            print("Compliance: {:2e}".format(comp))
            print("Time: {:.2f} [min]".format(t_total))
            print("Cost: {:.2f} [$]".format(c_total))
            print("Max Disp: {} [mm]".format(max_disp_val))
            print('comp2',comp2)

            # opt1.display_result_summary()
            # opt1.save_result()
        
#example: run_batch_opt("Studies/bracket/request_header.json","Studies/bracket","Studies/bracket/bc.json","Resources/mat_lib.json",'Resources/machine.json')
def run_batch_opt(request_header_json, study_folder,bc_json,mat_lib_json,machine_json,run_all=False,probe=False):
    for folder_name in os.listdir(study_folder):
        config_path = os.path.join(study_folder, folder_name)
        if os.path.isdir(config_path) and folder_name.startswith("config"): #looking for config subfolder
            active = True
            for file_name in os.listdir(config_path):
                if file_name == "PartProcessPlans.json":
                    active= False
            if active or run_all:
                run_opt(request_header_json,os.path.join(config_path,"mmm.json"), bc_json, mat_lib_json, machine_json,probe)
