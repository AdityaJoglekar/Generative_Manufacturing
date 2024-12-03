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

        self.SV_initam = 0.0
        self.SV_maxam =  100.0
        self.SV_deltaam = (self.SV_maxam - self.SV_initam)/200.0
        self.SV_coeffam = self.SV_initam

        self.directory_path = directory_path

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


        self.penal_init = 2.0
        self.penal_max = 4.0
        self.penal_delta = 0.01

        self.log_compliance = []
        self.log_vf = []

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




    #initialize the FEA solver
    def init_mat(self, E0, Emin, nu,mat_name):
        self.mat_name = mat_name
        self.fea.init_mat(E0, Emin, nu)

    def init_bc(self, il, jl, kl, il_F, jl_F,  kl_F, iif, jf, kf):
        self.fea.init_bc(il, jl, kl, il_F, jl_F,  kl_F, iif, jf, kf)

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



    def model_loss(self, epoch):
        xPhys = self.rbnn(self.dlX)
        penal = min(self.penal_init + self.penal_delta * epoch,self.penal_max)
        self.fea.penal = penal
        c = self.fea.compliance_cp(tf.reshape(xPhys,[self.nely,self.nelx,self.nelz]))
        vf = tf.math.reduce_mean(xPhys)
        part_mass = tf.reduce_sum(xPhys) *self.lele**3 * self.m_density


        if epoch < 100:
            self.SV_coeffam = max(min(self.SV_coeffam + self.SV_deltaam, self.SV_maxam),self.SV_initam)
            # self.SV_coeffam = 10
        else:
            self.SV_coeffam =  min(self.SV_coeffam + (epoch/100)**3,100)


        loss = c/self.c_0 + self.SV_coeffam*(tf.maximum(0,(part_mass/self.mass_con-1.0)))**2

        self.log_compliance.append(c)
        self.log_vf.append(vf)
        self.xPhys_final_np = xPhys.numpy().reshape([self.nely,self.nelx,self.nelz])
        return loss

    def train_step(self, epoch):
        with tf.GradientTape(persistent=True) as model_tape:
            loss = self.model_loss(epoch)
        param = [self.weights1,self.kernel,self.bias1,self.bias2]
        grad = model_tape.gradient(loss,param)
        grad = [tf.clip_by_norm(g, 0.1) for g in grad]
        self.to_optimizer.apply_gradients(zip(grad, param))

    def fit(self, epochs):
        for epoch in range(epochs):
            self.total_epoch = self.total_epoch + 1
            self.train_step(self.total_epoch)
            print("Epoch: ", self.total_epoch)    
            # if epoch%50 == 49 and epoch>20 and self.plots:
            if epoch%10 == 0 and epoch>20 and self.plots:
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
        self.fea.penal = 1.0
        comp = float((self.fea.compliance_cp(tf.cast(tf.reshape(xPhys,[self.nely,self.nelx,self.nelz]),dtype=tf.float32))*self.fea.E0).numpy())
        # max_d = float(self.fea.max_disp(tf.reshape(xPhys,[self.nely,self.nelx,self.nelz]))*1000)
        mass = tf.reduce_sum(xPhys) *self.lele**3 * self.m_density
        if self.plots:
            try:
                plot_iso(self,xPhys,0.2)
            except:
                print('Failed Rendering')
        print("## Result Summary")
        print("VF: ",tf.reduce_mean(xPhys))
        print("Mass: {:.6f} [kg]".format(mass))
        print("Compliance: {:2e}".format(comp))
        # print("Max Displacement: {:.2f}[mm]".format(max_d))

    def cutoff_calc(self,xPhys):
        cutoff_list = list(np.linspace(0.1,0.9,50))
        cutoff_eval_list = np.array([(np.sum(1.0*(xPhys>i)) - np.sum(xPhys)) for i in cutoff_list])
        cutoff_ind = min(len(cutoff_list)-1,int((max((np.array((np.where(cutoff_eval_list>0))).reshape(-1)).tolist()))))
        cutoff = cutoff_list[int(cutoff_ind)]
        return cutoff

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
            plt.margins(0,0,0)
            plt.savefig(os.path.join(self.directory_path, 'xPhys_iso.png'), bbox_inches = 'tight', pad_inches = 0, transparent=True)
            xPhysc = tf.convert_to_tensor(1.0*(xPhys>cutoff),dtype = tf.float32)
        except:
            print('Failed Rendering')

        # print('penalty: ',self.fea.penal)
        comp = float((self.fea.compliance_cp(tf.cast(tf.reshape(xPhys,[self.nely,self.nelx,self.nelz]),dtype=tf.float32))*self.fea.E0).numpy())
        mass = tf.reduce_sum(xPhys) *self.lele**3 * self.m_density
        self.data_dict['Mass (g)'].append(mass.numpy()*1000)
        self.data_dict['Compliance (Nm)'].append(comp)
        print('comp: ',comp)

def initialize_comp_and_vfs_in_objective(opt1, mass_con, mass_vf):
    opt1.mass_con = mass_con
    opt1.init_vf = mass_vf
    if opt1.init_vf<1.0:
        opt1.offset = tf.math.log(opt1.init_vf/(1-opt1.init_vf))
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
        

def run_opt(mmm_json = 'mmm.json',bc_json = 'bc.json',mat_lib_json = 'mat_lib.json', constraints_json = 'constraints.json', test = False, plots = True):

    directory_path = os.path.dirname(mmm_json)
    mmm = json.load(open(mmm_json))
    bc = json.load(open(bc_json))

    mat_lib = json.load(open(mat_lib_json))

    constraints = json.load(open(constraints_json))
    mass_con = constraints["Mass_constraint"][0]

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
    opt1.m_density = mat_lib[mmm["Material"]]["density"]
    opt1.mat_lib_file = mat_lib
    opt1.mmm_file = mmm
    opt1.plots = plots

    opt1.data_dict = collections.defaultdict(list)
    opt1.data_dict['Mass Constraint (g)'].append(mass_con*1000)
    opt1.data_dict['Manufacturing Method'].append('Unrestricted')



    #Initialize FEA solver
    opt1.fea.E0 = mat_lib[mmm["Material"]]["E0"]
    opt1.fea.KE =  lk_H8_np(nu)*opt1.lele
    opt1.init_bc(il, jl, kl, il_F, jl_F,  kl_F, iif, jf, kf)

    feasible = True
    #Calculate constraint dependent volume fractions
    curr_nele_m = mass_con/(opt1.lele**3 * opt1.m_density)
    mass_vf = curr_nele_m/opt1.nele
    if mass_vf>=1.0:
        mass_vf = 0.99999
    print('mass_vf: ',mass_vf)
    initialize_comp_and_vfs_in_objective(opt1, mass_con, mass_vf)

    if not test:
        #Infeasible constraints
        if mass_vf<0.02:
            print('mass_vf: ',mass_vf)
            # print(True)
            opt1.data_dict['Mass (g)'].append("infeasible")
            opt1.data_dict['Compliance (Nm)'].append("infeasible")
            opt1.data_dict['Material'].append(mmm["Material"])
            design_ai_data = pd.DataFrame(data = opt1.data_dict)
            design_ai_data.to_csv(os.path.join(directory_path, 'final_results.csv'),index = False)
            mmm["Valid"] = False
            mmm["Reason"] = "Infeasible Constraints"
            with open(os.path.join(opt1.directory_path,"mmm.json"),"w") as json_file:
                json.dump(mmm,json_file, indent=4)
            feasible = False

        else:
            print('mass_vf: ',mass_vf)
            #If constraints are too lenient, block of boundary condition envelope is optimal in terms of compliance 
            if (mass_vf>=0.99999):
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
                opt1.fea.penal = 1.0
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
            opt1.fea.penal = 1.0
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
            plot_iso2(xPhys,cutoff,-90,0, save = True)
            plt.margins(0,0,0)
            plt.savefig(os.path.join(opt1.directory_path, 'xPhys_iso_top.png'), bbox_inches = 'tight', pad_inches = 0, transparent=True)
            # plot_iso(opt1,xPhys,cutoff,save = True)
            # plot_iso2(xPhys,cutoff,210,120)
            # plot_iso2(xPhys,cutoff,210,-120)
            # plt.savefig(os.path.join(opt1.directory_path, 'xPhys_iso_top.png'), transparent=True)
            # plot_iso2(xPhys,cutoff,-90,0) 
            # plot_iso2(xPhys,cutoff,150,-120) 
            opt1.fea.penal = 1.0
            comp = opt1.fea.compliance_cp(tf.reshape(xPhys,[opt1.nely,opt1.nelx,opt1.nelz]))*opt1.fea.E0
            mass = tf.reduce_sum(xPhys) *opt1.lele**3 * opt1.m_density

            # final_df = pd.read_csv(os.path.join(opt1.directory_path,'final_results.csv'))
            # final_df['Compliance (Nm)'] = float(comp)
            # final_df['Maximum Displacement (mm)'] = max_disp_val
            # final_df.to_csv(os.path.join(directory_path, 'final_results.csv'),index = False)

            print("## Result Summary")
            print("VF: ",tf.reduce_mean(xPhys))
            print("Mass: {:.6f} [kg]".format(mass))
            print("Compliance: {:2e}".format(comp))
            # print('comp2',comp2)
            # print("Max Disp2: {} [mm]".format(max_disp_val2))

            # opt1.display_result_summary()
            # opt1.save_result()
        
