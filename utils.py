import tensorflow as tf
import os
from matplotlib import pyplot as plt
from matplotlib import cm, colors
from skimage import measure
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def plot_params(opt):
        plt.figure(figsize=(20,5))
        #VF log 
        plt.subplot(1,4,1)
        plt.title('Volume Fraction vs Iteration')
        plt.xlabel('iteration')
        plt.ylabel('VF')
        plt.plot(opt.log_vf)

        #Compliance log 
        plt.subplot(1,4,2)
        plt.title('Compliance vs Iteration')
        plt.xlabel('iteration')
        plt.ylabel('Compliance')
        plt.plot(opt.log_compliance)

        #Time log 
        plt.subplot(1,4,3)
        plt.title('Time vs Iteration')
        plt.xlabel('iteration')
        plt.ylabel('Time')
        plt.plot(opt.log_time)   

        #Cost log 
        plt.subplot(1,4,4)
        plt.title('Cost vs Iteration')
        plt.xlabel('iteration')
        plt.ylabel('Cost')
        plt.plot(opt.log_cost)
        plt.show() 

def display_seg(opt, xPhys):
    plt.figure(figsize=(15,15))
    # combined vf plot
    plt.subplot(1,1+1,1)
    plt.title('Combined')
    xPhys = tf.reshape(xPhys,[opt.nely,opt.nelx,opt.nelz])
    plt.imshow(tf.reduce_mean(xPhys,axis=2),vmin=0, vmax=1,cmap = 'seismic')
    plt.show()

def plot3d(xPhys, save=False): #xPhys in 3D shape
    cutoff = 0.3

    xPhys = np.flip(xPhys,axis=2)
    voxelarray = np.copy(xPhys)
    voxelarray[np.where(voxelarray<cutoff)] = 0
    voxelarray[np.where(voxelarray>cutoff)] = 1
    voxelarray.astype(bool)
    voxelarray = np.flip(voxelarray,axis=0)
    ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')
    norm = colors.Normalize(vmin=0.0, vmax=1.0, clip=False)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.seismic)

    color = mapper.to_rgba((xPhys.reshape([-1]))).reshape([xPhys.shape[0],xPhys.shape[1],xPhys.shape[2],4])
    color = np.flip(color,axis=0)
    ax.voxels(voxelarray,facecolors=color, edgecolor='k')
    ax.view_init(20, -60,vertical_axis='x')
    ax.set_box_aspect(aspect = (xPhys.shape[1],xPhys.shape[2],xPhys.shape[0]))
    ax.set_axis_off()
    if not save:
        plt.show()

def plot_iso(opt, xPhys,cutoff,save=False ): #xPhys in shape [nely,nelx,nelz] and numpy array
    nely, nelx, nelz = xPhys.shape
    nelm = max(nely,nelx,nelz)
    padding = np.zeros([nely+2,nelx+2,nelz+2])
    padding[1:-1, 1:-1, 1:-1] = np.copy(xPhys)
    xPhys = padding
    verts, faces, normals, values = measure.marching_cubes(xPhys, cutoff) #set the density cutoff
    fig = plt.figure(figsize=(2, 2), dpi = 300)
    ax = fig.add_subplot(1,1,1,projection='3d')
    ls = LightSource(-45, 0)
    f_coord = np.take(verts, faces,axis = 0)
    f_norm = np.cross(f_coord[:,2] - f_coord[:,0], f_coord[:,1] - f_coord[:,0])
    cl = ls.shade_normals(f_norm)
    norm = colors.Normalize(vmin=0.0, vmax=1.0, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.gray)
    rgb = mapper.to_rgba(cl).reshape([-1,4])
    mesh = Poly3DCollection(np.take(verts, faces,axis = 0)/nelx * (np.array([[nelm/nely, nelm/nelm, nelm/nelz]])))
    #mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)
    mesh.set_facecolors(rgb)
    ax.view_init(210, -120,vertical_axis='x')
    ax.set_box_aspect(aspect = (nelx,nelz,nely))
    ax.set_axis_off()
    if save:
        xPhysr = np.rot90(xPhys, k=3,axes = (0,1))
        verts, faces, normals, values = measure.marching_cubes(xPhysr, level = cutoff,spacing = (100*opt.lele,100*opt.lele,100*opt.lele)) #set the density cutoff
        faces = faces + 1
        with open(os.path.join(opt.directory_path,"xPhys_iso.obj"), 'w') as file:
            for item in verts:
                file.write(f"v {item[0]} {item[1]} {item[2]}\n")

            for item in normals:
                file.write(f"vn {-item[0]} {-item[1]} {-item[2]}\n")

            for item in faces:
                file.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0],item[1],item[2]))
    else:
        plt.show()

def plot_iso2(xPhys, cutoff, or1, or2, save=False ): #xPhys in shape [nely,nelx,nelz] and numpy array
    nely, nelx, nelz = xPhys.shape
    nelm = max(nely,nelx,nelz)
    padding = np.zeros([nely+2,nelx+2,nelz+2])
    padding[1:-1, 1:-1, 1:-1] = np.copy(xPhys)
    xPhys = padding
    verts, faces, normals, values = measure.marching_cubes(xPhys, cutoff) #set the density cutoff
    fig = plt.figure(figsize=(2, 2), dpi = 300)
    ax = fig.add_subplot(1,1,1,projection='3d')
    ls = LightSource(-45, 0)
    f_coord = np.take(verts, faces,axis = 0)
    f_norm = np.cross(f_coord[:,2] - f_coord[:,0], f_coord[:,1] - f_coord[:,0])
    cl = ls.shade_normals(f_norm)
    norm = colors.Normalize(vmin=0.0, vmax=1.0, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.gray)
    rgb = mapper.to_rgba(cl).reshape([-1,4])
    mesh = Poly3DCollection(np.take(verts, faces,axis = 0)/nelx * (np.array([[nelm/nely, nelm/nelm, nelm/nelz]])))
    #mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)
    mesh.set_facecolors(rgb)
    ax.view_init(or1, or2,vertical_axis='x')
    ax.set_box_aspect(aspect = (nelx,nelz,nely))
    ax.set_axis_off()
    if not save:
        plt.show()