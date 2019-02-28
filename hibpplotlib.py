# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:01:34 2019

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from mpl_toolkits import mplot3d
import visvis as vv
import wire
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from hibplib import *

# %%
'''############################################################################
Functions for plotting Mafnetic field '''
# %% visvis plot 3D
def plot_3d(B, wires, cutoff=2):
    '''
    plot absolute values of B in 3d with visvis
    :param B: magnetic field values array (has 3 dimensions) [T]
    :param wires: list of wire objects
    :return: None
    '''

    Babs = np.linalg.norm(B, axis=1)
    Babs[Babs > cutoff] = np.nan

    # draw results
    # prepare axes
    a = vv.gca()
    a.cameraType = '3d'
    a.daspectAuto = False

    vol = Babs.reshape(grid.shape[1:]).T
    vol = vv.Aarray(vol, sampling=(resolution, resolution, resolution),
                    origin=(volume_corner1[2], volume_corner1[1],
                            volume_corner1[0]))
    # set labels
    vv.xlabel('x axis')
    vv.ylabel('y axis')
    vv.zlabel('z axis')

    wire.vv_PlotWires(wires)

    t = vv.volshow2(vol, renderStyle='mip', cm=vv.CM_JET)
    vv.colorbar()
    app = vv.use()
    app.Run()

# %% matplotlib plot 2D
def plot_2d(B, points, plane='xy', cutoff=2, n_contours=50):
    '''
    make contour plot of B in XZ and XY plane
    :param B: magnetic field values array (has 3 dimensions) [T]
    :param points: coordinates for points for B vectors to start on (@My english is perfect)
    :return: None
    '''
    pf_coils = importPFCoils('PFCoils.dat')
    #2d quiver
    # get 2D values from one plane with Y = 0
    fig = plt.figure()
    ax = fig.gca()
    if plane == 'xz':
        # choose y position
        mask = (np.around(points[:, 1], 3) == 0.1)
        B = B[mask]
        Babs = np.linalg.norm(B, axis=1)
        B[Babs > cutoff] = [np.nan,np.nan,np.nan]
        points = points[mask]
#        ax.quiver(points[:, 0], points[:, 2], B[:, 0], B[:, 2], scale=2.0)

        X = np.unique(points[:, 0])
        Z = np.unique(points[:, 2])
        cs = ax.contour(X, Z, Babs.reshape([len(X), len(Z)]).T, n_contours)
        ax.clabel(cs)
        plt.xlabel('x')
        plt.ylabel('z')

    elif plane == 'xy':
        # get coil inner and outer profile
        filename = 'coildata.dat'
        array = np.loadtxt(filename) # go to [m]
        # array has only x and y columns, so soon we need to add a zero column for z
        # (because we have no z column in coildata)
        outer_coil = np.array(array[:, [2, 3]])
        inner_coil = np.array(array[:, [0, 1]])
        #plot toroidal coil
        plt.plot(inner_coil[:, 0], inner_coil[:, 1], color='k')
        plt.plot(outer_coil[:, 0], outer_coil[:, 1], color='k')

        #plot pf coils
        for coil in pf_coils.keys():
            xc = pf_coils[coil][0]
            yc = pf_coils[coil][1]
            dx = pf_coils[coil][2]
            dy = pf_coils[coil][3]
            ax.add_patch(Rectangle((xc-dx/2, yc-dy/2), dx, dy,
                                   linewidth=1, edgecolor='r', facecolor='r'))
        # choose z position
        mask = (np.around(points[:, 2], 3) == 0.05)
        B = B[mask]
        Babs = np.linalg.norm(B, axis=1)
        B[Babs > cutoff] = [np.nan, np.nan, np.nan]
        points = points[mask]
        Babs = np.linalg.norm(B, axis=1)
        ax.quiver(points[:, 0], points[:, 1], B[:, 0], B[:, 1], scale=20.0)

        X = np.unique(points[:, 0])
        Y = np.unique(points[:, 1])
        cs = ax.contour(X, Y, Babs.reshape([len(X), len(Y)]).T, n_contours)
        ax.clabel(cs)

        plt.xlabel('x')
        plt.ylabel('y')

    plt.axis('equal')
    plt.show()

# %% matplotlib plot 3D
def plot_3dm(B, wires, points, cutoff=2):
    '''
    plot 3d quiver of B using matplotlib
    '''
    Babs = np.linalg.norm(B, axis=1)
    B[Babs > cutoff] = [np.nan, np.nan, np.nan]

    fig = plt.figure()
    # 3d quiver
    ax = fig.gca(projection='3d')
    wire.mpl3d_PlotWires(wires, ax)
#    ax.quiver(points[:, 0], points[:, 1], points[:, 2],
#              B[:, 0], B[:, 1], B[:, 2], length=20)
    plt.show()

# %%
'''############################################################################
Functions for plotting electric field'''
# %%
def plot_contours(X, Y, Z, U, n_contours=30):
    '''
    contour plot of potential U
    :param X, Y, Z: mesh ranges in X, Y and Z respectively [m]
    :param U:  plate's U  [V]
    :param n_contours:  number of planes to skip before plotting
    :return: None
    '''
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    ax1.contour(X, Y, U[:,:,U.shape[2]//2].swapaxes(0, 1), n_contours)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.grid(True)
    ax1.axis('equal')
    # add the edge of the domain
    domain = patches.Rectangle((min(X), min(Y)), max(X)-min(X), max(Y)-min(Y),
                               linewidth=2, linestyle='--', edgecolor='k',
                               facecolor='none')
    ax1.add_patch(domain)

    ax2.contour(Z, Y, U[U.shape[0]//2, :, :], n_contours)
    ax2.set_xlabel('Z (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True)
    ax2.axis('equal')
    # add the edge of the domain
    domain = patches.Rectangle((min(Z), min(Y)), max(Z)-min(Z), max(Y)-min(Y),
                               linewidth=2, linestyle='--', edgecolor='k',
                               facecolor='none')
    ax2.add_patch(domain)

# %%
def plot_quiver(X, Y, Z, Ex, Ey, Ez):
    '''
    quiver plot of Electric field in xy, xz, zy planes
    :param X, Y, Z: mesh ranges in X, Y and Z respectively [m]
    :param Ex, Ey, Ez:  plate's U gradient components [V/m]
    :param n_skip:  number of planes to skip before plotting
    :return: None
    '''
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

    z_cut = Ex.shape[2]//2 + 20 # z position of XY cut
    ax1.quiver(X, Y, Ex[:, :, z_cut].swapaxes(0, 1),
                     Ey[:, :, z_cut].swapaxes(0, 1))
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.grid(True)
    ax1.axis('equal')
    # add the edge of the domain
    domain = patches.Rectangle((min(X), min(Y)), max(X)-min(X), max(Y)-min(Y),
                               linewidth=2, linestyle='--', edgecolor='k',
                               facecolor='none')
    ax1.add_patch(domain)

    x_cut = Ey.shape[0]//2 # x position of ZY cut
    ax2.quiver(Z, Y, Ez[x_cut, :, :],
                     Ey[x_cut, :, :])
    ax2.set_xlabel('Z (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True)
    ax2.axis('equal')
    # add the edge of the domain
    domain = patches.Rectangle((min(Z), min(Y)), max(Z)-min(Z), max(Y)-min(Y),
                               linewidth=2, linestyle='--', edgecolor='k',
                               facecolor='none')
    ax2.add_patch(domain)

    y_cut = Ex.shape[1]//2 # y position of XZ cut
    ax3.quiver(X, Z, Ex[:, y_cut, :].swapaxes(0, 1),
                     Ez[:, y_cut, :].swapaxes(0, 1))
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.grid(True)
    ax3.axis('equal')
    # add the edge of the domain
    domain = patches.Rectangle((min(X), min(Z)), max(X)-min(X), max(Z)-min(Z),
                               linewidth=2, linestyle='--', edgecolor='k',
                               facecolor='none')
    ax3.add_patch(domain)

# %%
def plot_quiver3d(X, Y, Z, Ex, Ey, Ez, n_skip=5):
    '''
    3d quiver plot of Electric field
    :param X, Y, Z: mesh ranges in X, Y and Z respectively
    :param Ex, Ey, Ez:  plate's U gradient components
    :param n_skip:  number of planes to skip before plotting
    :return: None
    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #plot plates (very scematically)
    ax.plot(UP_rotated[:, 0],
            UP_rotated[:, 1], UP_rotated[:, 2],'-o', color='b')
    ax.plot(LP_rotated[:, 0],
            LP_rotated[:, 1], LP_rotated[:, 2],'-o', color='r')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.grid(True)

    x_pos = X.shape[0]//2
    y_pos = Y.shape[1]//2
    z_pos = Z.shape[2]//2
#    skip = (slice(None, None, n_skip), y_pos, slice(None,None,n_skip))
    skip = (x_pos, slice(None, None, 3*n_skip), slice(None, None, n_skip))
    ax.quiver3D(X[skip], Y[skip], Z[skip],
                Ex[skip],
                Ey[skip],
                Ez[skip], length=0.01, normalize=True)

    skip = (slice(None, None, 3*n_skip), y_pos, slice(None, None, n_skip))
    ax.quiver3D(X[skip], Y[skip], Z[skip],
                Ex[skip],
                Ey[skip],
                Ez[skip], length=0.01, normalize=True)

    ax.axis('equal')

# %%
'''############################################################################
Functions for plotting trajectories'''

def plot_geometry(ax):
    '''
    plot toroidal and poloidal field coils, camera and separatrix
    :param ax: graph to plot geometry on
    :return: None
    '''
    # get T-15 coil inner and outer profile
    filename = 'coildata.dat'
    array = np.loadtxt(filename)  # [m]
    #array has only x and y columns, so soon we need to add a zero column for z
    # (because we have no z column in coildata)
    outer_coil = np.array(array[:, [2, 3]])
    inner_coil = np.array(array[:, [0, 1]])
    #plot toroidal coil
    ax.plot(inner_coil[:, 0], inner_coil[:, 1], '--', color='k')
    ax.plot(outer_coil[:, 0], outer_coil[:, 1], '--', color='k')

    #get T-15 camera and plasma contours
    filename = 'T15_vessel.txt'
    camera = np.loadtxt(filename)/1000
    filename = 'T15_sep.txt'
    separatrix = np.loadtxt(filename)/1000
    ax.plot(camera[:, 0]+1.5, camera[:, 1], color='tab:blue')
    ax.plot(separatrix[:, 0]+1.5, separatrix[:, 1], color='tab:orange')

    pf_coils = importPFCoils('PFCoils.dat')

    #plot pf coils
    for coil in pf_coils.keys():
        xc = pf_coils[coil][0]
        yc = pf_coils[coil][1]
        dx = pf_coils[coil][2]
        dy = pf_coils[coil][3]
        ax.add_patch(Rectangle((xc-dx/2, yc-dy/2), dx, dy,
                               linewidth=1, edgecolor='tab:gray', facecolor='tab:gray'))

# %%
def easy_plot(tr, r_aim):
    '''
    plotting calculated trajectories in XY plane
    :param tr: trajectory to plot
    :param r_aim: aim dot coordinates [m]
    :return: None
    '''
    fig = plt.figure()
    ax = fig.gca()

    plot_geometry(ax)

    # plot trajectories
    plt.plot(tr.RV_Prim[:,0], tr.RV_Prim[:,1], 'o', color='k', \
             label = 'E = {} keV, alf = {}, bet = {}'.format(tr.Ebeam, tr.alpha, tr.beta))
    plt.plot(tr.RV_Sec[:,0], tr.RV_Sec[:,1], 'o', color='r')

    plt.plot(r_aim[0,0],r_aim[0,1],'*')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True)
    plt.title('E={} keV'.format(tr.Ebeam))
    plt.axis('equal')

def easy_plotXZ(tr, r_aim):
    '''
    plotting calculated trajectories in XY plane
    :param tr: trajectory to plot
    :param r_aim: aim dot coordinates [m]
    :return: None
    '''
    plt.figure()

    # plot trajectories
    plt.plot(tr.RV_Prim[:,0], tr.RV_Prim[:,2], '-', color='k')
    plt.plot(tr.RV_Sec[:,0], tr.RV_Sec[:,2], '-', color='r')

    plt.plot(r_aim[0,0],r_aim[0,1],'*')

    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.grid(True)

    plt.axis('equal')

#%%
def plot_fan(traj_list, r_aim, A2_edges, B2_edges,
             Ebeam, UA2, Btor, Ipl,
             tick_width=2, axis_labelsize=18, title_labelsize=18):
    '''
    plot fan of trajectories in xy, xz and zy planes
    :param traj_list: list of trajectories
    :param r_aim: aim dot coordinates [m]
    :param A2_edges: A2 edges coordinates [m]
    :param B2_edges: B2 edges coordinates [m]
    :param Ebeam: beam energy [keV]
    :param Btor: toroidal magnetic field [T]
    :param Ipl: plasma current [A]
    :return: None
    '''

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

    # Grids
    ax1.grid(True)
    ax1.grid(which='major', color = 'tab:gray') #draw primary grid
    ax1.minorticks_on() # make secondary ticks on axes
    ax1.grid(which='minor', color = 'tab:gray', linestyle = ':') # draw secondary grid

    ax2.grid(True)
    ax2.grid(which='major', color = 'tab:gray') #draw primary grid
    ax2.minorticks_on() # make secondary ticks on axes
    ax2.grid(which='minor', color = 'tab:gray', linestyle = ':') # draw secondary grid

    ax3.grid(True)
    ax3.grid(which='major', color = 'tab:gray') #draw primary grid
    ax3.minorticks_on() # make secondary ticks on axes
    ax3.grid(which='minor', color = 'tab:gray', linestyle = ':') # draw secondary grid

    # Axis

    ax1.xaxis.set_tick_params(width=tick_width) # increase tick size
    ax1.yaxis.set_tick_params(width=tick_width)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.axis('equal')

    ax2.xaxis.set_tick_params(width=tick_width) # increase tick size
    ax2.yaxis.set_tick_params(width=tick_width)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.axis('equal')

    ax3.xaxis.set_tick_params(width=tick_width) # increase tick size
    ax3.yaxis.set_tick_params(width=tick_width)
    ax3.set_xlabel('Z (m)')
    ax3.set_ylabel('Y (m)')
    ax3.axis('equal')

    # get T-15 camera and plasma contours
    plot_geometry(ax1)

    # plot aim dot
    ax1.plot(r_aim[0,0],r_aim[0,1],'*')
    ax2.plot(r_aim[0,0],r_aim[0,2],'*')
    ax3.plot(r_aim[0,2],r_aim[0,1],'*')

    for tr in traj_list:
        if tr.Ebeam == Ebeam and tr.UA2 == UA2:
            #plot plates
            ax1.plot(A2_edges[0][[0,3],0],A2_edges[0][[0,3],1], 
                     color='k', linewidth = 2)
            ax1.plot(A2_edges[1][[0,3],0],A2_edges[1][[0,3],1], 
                     color='k', linewidth = 2)
            ax1.fill(B2_edges[0][:,0], B2_edges[0][:,1], fill=False, 
                     hatch='//', linewidth = 2)

            ax1.plot(tr.RV_Prim[:,0], tr.RV_Prim[:,1],color='k')
            ax2.plot(tr.RV_Prim[:,0], tr.RV_Prim[:,2],color='k')
            ax3.plot(tr.RV_Prim[:,2], tr.RV_Prim[:,1],color='k')

            last_points = []
            for i in tr.Fan:
                ax1.plot(i[:,0], i[:,1],color='r')
                ax2.plot(i[:,0], i[:,2],color='r')
                ax3.plot(i[:,2], i[:,1],color='r')
                last_points.append(i[-1, :])
            last_points = np.array(last_points)
            ax3.plot(last_points[:, 2], last_points[:, 1], '--o', color='r')

    ax1.set_title('E={} keV, UA2={} kV, Btor = {} T, Ipl = {} MA'.format(tr.Ebeam,tr.UA2, Btor, Ipl))

#%%
def plot_fan_xy(traj_list, r_aim, A2_edges, B2_edges,
             Ebeam, UA2, Btor, Ipl,
             tick_width=2, axis_labelsize=18, title_labelsize=18):
    '''
    plot fan of trajectories in xy
    :param traj_list: list of trajectories
    :param r_aim: aim dot coordinates [m]
    :param A2_edges: A2 edges coordinates [m]
    :param B2_edges: B2 edges coordinates [m]
    :param Ebeam: beam energy [keV]
    :param Btor: toroidal magnetic field [T]
    :param Ipl: plasma current [A]
    :return: None
    '''
    fig, ax1 = plt.subplots()

    # Grids
    ax1.grid(True)
    ax1.grid(which='major', color = 'tab:gray') #draw primary grid
    ax1.minorticks_on() # make secondary ticks on axes
    ax1.grid(which='minor', color = 'tab:gray', linestyle = ':') # draw secondary grid

    # Axis
    ax1.xaxis.set_tick_params(width=tick_width) # increase tick size
    ax1.yaxis.set_tick_params(width=tick_width)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.axis('equal')

    # get T-15 camera and plasma contours
    plot_geometry(ax1)

    # plot aim dot
    ax1.plot(r_aim[0,0],r_aim[0,1],'*')

    for tr in traj_list:
        if tr.Ebeam == Ebeam and tr.UA2 == UA2:
            #plot plates
            ax1.plot(A2_edges[0][[0,3],0],A2_edges[0][[0,3],1],  color='k', linewidth = 2)
            ax1.plot(A2_edges[1][[0,3],0],A2_edges[1][[0,3],1],  color='k', linewidth = 2)
            ax1.fill(B2_edges[0][:,0], B2_edges[0][:,1], fill=False, hatch='//', linewidth = 2)

            ax1.plot(tr.RV_Prim[:,0], tr.RV_Prim[:,1],color='k')

            last_points = []
            for i in tr.Fan:
                ax1.plot(i[:,0], i[:,1],color='r')
                last_points.append(i[-1, :])
            last_points = np.array(last_points)

    ax1.set_title('E={} keV, UA2={} kV, Btor = {} T, Ipl = {} MA'.format(tr.Ebeam,tr.UA2, Btor, Ipl))

# %%
def plot_scan(traj_list, r_aim,
              Ebeam, Btor, Ipl,
              tick_width=2, axis_labelsize=18, title_labelsize=18):

    '''
    plot scan for one beam with particular energy in 2 planes: xy, xz
    :param traj_list: list of trajectories
    :param r_aim: aim dot coordinates [m]
    :param Ebeam: beam energy [keV]
    :param Btor: toroidal magnetic field [T]
    :param Ipl: plasma current [A]
    :return: None
    '''

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    # Grids
    ax1.grid(True)
    ax1.grid(which='major', color = 'tab:gray') #draw primary grid
    ax1.minorticks_on() # make secondary ticks on axes
    ax1.grid(which='minor', color = 'tab:gray', linestyle = ':') # draw secondary grid

    ax2.grid(True)
    ax2.grid(which='major', color = 'tab:gray') #draw primary grid
    ax2.minorticks_on() # make secondary ticks on axes
    ax2.grid(which='minor', color = 'tab:gray', linestyle = ':') # draw secondary grid

    # Axis

    ax1.xaxis.set_tick_params(width=tick_width) # increase tick size
    ax1.yaxis.set_tick_params(width=tick_width)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.axis('equal')

    ax2.xaxis.set_tick_params(width=tick_width) # increase tick size
    ax2.yaxis.set_tick_params(width=tick_width)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.axis('equal')

    # get T-15 camera and plasma contours
    plot_geometry(ax1)
    # plot aim dot
    ax1.plot(r_aim[0,0],r_aim[0,1],'*')
    ax2.plot(r_aim[0,0],r_aim[0,2],'*')

    # get the list of UA2
    A2list = []
    for i in range(len(traj_list)):
        A2list.append(traj_list[i].UA2)

    #find UA2 max and min
    UA2_max = np.amax(np.array(A2list))
    UA2_min = np.amin(np.array(A2list))

    ax1.set_title('Ebeam={} keV, UA2:[{}, {}] Kv, Btor = {} T, Ipl = {} MA'
          .format(Ebeam, UA2_min,  UA2_max,
                  Btor, Ipl))

    for i in range(len(traj_list)):
        if traj_list[i].Ebeam == Ebeam:
            # plot trajectories
            ax1.plot(traj_list[i].RV_Prim[:,0], traj_list[i].RV_Prim[:,1], color='k')
            ax1.plot(traj_list[i].RV_Sec[:,0], traj_list[i].RV_Sec[:,1], color='r')

            ax2.plot(traj_list[i].RV_Prim[:,0], traj_list[i].RV_Prim[:,2], color='k')
            ax2.plot(traj_list[i].RV_Sec[:,0], traj_list[i].RV_Sec[:,2], color='r')


# %%
def plot_scan_xy(traj_list, r_aim,
              Ebeam, Btor, Ipl,
              tick_width=2, axis_labelsize=18, title_labelsize=18):

    '''
    plot scan for one beam with particular energy in 2 planes: xy, xz
    :param traj_list: list of trajectories
    :param r_aim: aim dot coordinates [m]
    :param Ebeam: beam energy [keV]
    :param Btor: toroidal magnetic field [T]
    :param Ipl: plasma current [A]
    :return: None
    '''

    fig, ax1 = plt.subplots()

    # Grids
    ax1.grid(True)
    ax1.grid(which='major', color = 'tab:gray') #draw primary grid
    ax1.minorticks_on() # make secondary ticks on axes
    ax1.grid(which='minor', color = 'tab:gray', linestyle = ':') # draw secondary grid

    # Axis
    ax1.xaxis.set_tick_params(width=tick_width) # increase tick size
    ax1.yaxis.set_tick_params(width=tick_width)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.axis('equal')

    # get T-15 camera and plasma contours
    plot_geometry(ax1)
    # plot aim dot
    ax1.plot(r_aim[0,0],r_aim[0,1],'*')

    # get the list of UA2
    A2list = []
    for i in range(len(traj_list)):
        A2list.append(traj_list[i].UA2)

    #find UA2 max and min
    UA2_max = np.amax(np.array(A2list))
    UA2_min = np.amin(np.array(A2list))

    ax1.set_title('Ebeam={} keV, UA2:[{}, {}] Kv, Btor = {} T, Ipl = {} MA'
          .format(Ebeam, UA2_min,  UA2_max,
                  Btor, Ipl))

    for i in range(len(traj_list)):
        if traj_list[i].Ebeam == Ebeam:
            # plot trajectories
            ax1.plot(traj_list[i].RV_Prim[:,0], traj_list[i].RV_Prim[:,1], color='k')
            ax1.plot(traj_list[i].RV_Sec[:,0], traj_list[i].RV_Sec[:,1], color='r')

# %%
def plot_grid(traj_list, r_aim, Btor, Ipl,
              tick_width=2, axis_labelsize=18, title_labelsize=18,
              linestyle_A2='--', linestyle_E='-',
              marker_A2='*', marker_E='p',
              traj_color='tab:gray'):
    '''
    plot detector grid in 2 planes: xy, xz
    :param traj_list: list of trajectories
    :param r_aim: aim dot coordinates [m]
    :param Btor: toroidal magnetic field [T]
    :param Ipl: plasma current [A]
    :return: None
    '''

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    # Grids
    ax1.grid(True)
    ax1.grid(which='major', color = 'tab:gray') #draw primary grid
    ax1.minorticks_on() # make secondary ticks on axes
    ax1.grid(which='minor', color = 'tab:gray', linestyle = ':') # draw secondary grid

    ax2.grid(True)
    ax2.grid(which='major', color = 'tab:gray') #draw primary grid
    ax2.minorticks_on() # make secondary ticks on axes
    ax2.grid(which='minor', color = 'tab:gray', linestyle = ':') # draw secondary grid

    # Axis
    ax1.xaxis.set_tick_params(width=tick_width) # increase tick size
    ax1.yaxis.set_tick_params(width=tick_width)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.axis('equal')

    ax2.xaxis.set_tick_params(width=tick_width) # increase tick size
    ax2.yaxis.set_tick_params(width=tick_width)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.axis('equal')

    # get T-15 camera and plasma contours
    plot_geometry(ax1)

    # get the list of A2 and Ebeam
    A2list = []
    Elist = []
    for i in range(len(traj_list)):
        A2list.append(traj_list[i].UA2)
        Elist.append(traj_list[i].Ebeam)

    # make sorted arrays of non repeated values
    A2list = np.unique(A2list)
    N_A2 = A2list.shape[0]
    Elist = np.unique(Elist)
    N_E = Elist.shape[0]

    E_grid = np.zeros((N_A2,3,N_E))
    E_grid[:] = np.nan
    A2_grid = np.zeros((N_E,3,N_A2))
    A2_grid[:] = np.nan

    # find UA2 max and min
    UA2_max = np.amax(np.array(A2list))
    UA2_min = np.amin(np.array(A2list))

    ax1.set_title('Ebeam:[{}, {}] keV, UA2:[{}, {}] Kv, Btor = {} T, Ipl = {} kA'
                  .format(traj_list[0].Ebeam, traj_list[-1].Ebeam,
                         UA2_min,  UA2_max,
                          Btor, Ipl))

    # plot raim
    ax1.plot(r_aim[0,0],r_aim[0,1],'*')
    ax2.plot(r_aim[0,0],r_aim[0,2],'*')

    #make a grid of constant E
    for i_E in range(0,N_E,1):
        k = -1
        for i_tr in range(len(traj_list)):
            if  traj_list[i_tr].Ebeam == Elist[i_E]:
                k += 1
                #take the 1-st point of secondary trajectory
                x = traj_list[i_tr].RV_Sec[0,0]
                y = traj_list[i_tr].RV_Sec[0,1]
                z = traj_list[i_tr].RV_Sec[0,2]
                E_grid[k,:,i_E] = [x,y,z]

        ax1.plot(E_grid[:,0,i_E], E_grid[:,1,i_E],
                 linestyle=linestyle_E,
                 marker=marker_E,
                 label=str(int(Elist[i_E]))+' keV')
        ax2.plot(E_grid[:,0,i_E], E_grid[:,2,i_E],
                 linestyle=linestyle_E,
                 marker=marker_E,
                 label=str(int(Elist[i_E]))+' keV')

    #make a grid of constant A2
    for i_A2 in range(0,N_A2,1):
        k = -1
        for i_tr in range(len(traj_list)):
            if traj_list[i_tr].UA2 == A2list[i_A2]:
                k += 1
                #take the 1-st point of secondary trajectory
                x = traj_list[i_tr].RV_Sec[0,0]
                y = traj_list[i_tr].RV_Sec[0,1]
                z = traj_list[i_tr].RV_Sec[0,2]
                A2_grid[k,:,i_A2] = [x,y,z]


        ax1.plot(A2_grid[:,0,i_A2], A2_grid[:,1,i_A2],
                 linestyle=linestyle_A2,
                 marker=marker_A2,
                 label=str(round(A2list[i_A2],1))+' kV')
        ax2.plot(A2_grid[:,0,i_A2], A2_grid[:,2,i_A2],
                 linestyle=linestyle_A2,
                 marker=marker_A2,
                 label=str(round(A2list[i_A2],1))+' kV')

    ax1.legend()

#%%
def plot_grid_xy(traj_list, r_aim, Btor, Ipl, legend=True, linestyle_A2='--', 
                  linestyle_E='-', marker_A2='*', marker_E='p',
                  traj_color='tab:gray'):
    '''
    plot detector grid in 2 planes: xy, xz
    :param traj_list: list of trajectories
    :param r_aim: aim dot coordinates [m]
    :param Btor: toroidal magnetic field [T]
    :param Ipl: plasma current [A]
    :return: grid figure
    '''

    fig, ax1 = plt.subplots()

    # Grids
    ax1.grid(True)
    ax1.grid(which='major', color = 'tab:gray') #draw primary grid
    ax1.minorticks_on() # make secondary ticks on axes
    ax1.grid(which='minor', color = 'tab:gray', linestyle = ':') # draw secondary grid

    # Fonts and ticks
    plt.tick_params(axis='both', which='major', labelsize=18) # increase label font size
    ax1.xaxis.set_tick_params(width=2) # increase tick size
    ax1.yaxis.set_tick_params(width=2)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.axis('equal')

    #get T-15 camera and plasma contours
    plot_geometry(ax1)

    # get the list of A2 and Ebeam
    A2list = []
    Elist = []
    for i in range(len(traj_list)):
        A2list.append(traj_list[i].UA2)
        Elist.append(traj_list[i].Ebeam)

    #make sorted arrays of non repeated values
    A2list = np.unique(A2list)
    N_A2 = A2list.shape[0]
    Elist = np.unique(Elist)
    N_E = Elist.shape[0]

    E_grid = np.zeros((N_A2,3,N_E))
    E_grid[:] = np.nan
    A2_grid = np.zeros((N_E,3,N_A2))
    A2_grid[:] = np.nan

    #find UA2 max and min
    UA2_max = np.amax(np.array(A2list))
    UA2_min = np.amin(np.array(A2list))

    ax1.set_title('Ebeam:[{}, {}] keV, UA2:[{}, {}] Kv, Btor = {} T, Ipl = {} kA'
                  .format(traj_list[0].Ebeam, traj_list[-1].Ebeam,
                         UA2_min,  UA2_max,
                          Btor, Ipl))

    #make a grid of constant A2
    for i_A2 in range(0,N_A2,1):
        k = -1
        for i_tr in range(len(traj_list)):
            if traj_list[i_tr].UA2 == A2list[i_A2]:
                k += 1
                #take the 1-st point of secondary trajectory
                x = traj_list[i_tr].RV_Sec[0,0]
                y = traj_list[i_tr].RV_Sec[0,1]
                z = traj_list[i_tr].RV_Sec[0,2]
                A2_grid[k,:,i_A2] = [x,y,z]


        ax1.plot(A2_grid[:,0,i_A2], A2_grid[:,1,i_A2],
                 linestyle=linestyle_A2,
                 marker=marker_A2,
                 label=str(round(A2list[i_A2],1))+' kV')

    #make a grid of constant E
    for i_E in range(0,N_E,1):
        k = -1
        for i_tr in range(len(traj_list)):
            if traj_list[i_tr].Ebeam == Elist[i_E]:
                k += 1
                #take the 1-st point of secondary trajectory
                x = traj_list[i_tr].RV_Sec[0,0]
                y = traj_list[i_tr].RV_Sec[0,1]
                z = traj_list[i_tr].RV_Sec[0,2]
                E_grid[k,:,i_E] = [x,y,z]

        ax1.plot(E_grid[:,0,i_E], E_grid[:,1,i_E],
                 linestyle=linestyle_E,
                 marker=marker_E,
                 label=str(round(Elist[i_E],1))+' keV')


    ax1.plot(r_aim[0,0],r_aim[0,1],'*')


    if legend:
        ax1.legend()

#%%
def plot_geometry_xy(r_aim, rA2, rB2, A2_edges, B2_edges,
                     tick_width=2, axis_labelsize=18, title_labelsize=18):
    '''
    plot geometry configuration in xy
    :param traj_list: list of trajectories
    :param r_aim: aim dot coordinates [m]
    :param A2_edges: A2 edges coordinates [m]
    :param B2_edges: B2 edges coordinates [m]
    :param Ebeam: beam energy [keV]
    :param Btor: toroidal magnetic field [T]
    :param Ipl: plasma current [A]
    :return: None
    '''
    fig, ax1 = plt.subplots()

    # Grids
    ax1.grid(True)
    ax1.grid(which='major', color = 'tab:gray') #draw primary grid
    ax1.minorticks_on() # make secondary ticks on axes
    ax1.grid(which='minor', color = 'tab:gray', linestyle = ':') # draw secondary grid

    # Axis
    ax1.xaxis.set_tick_params(width=tick_width) # increase tick size
    ax1.yaxis.set_tick_params(width=tick_width)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.axis('equal')

    # get T-15 camera and plasma contours
    plot_geometry(ax1)

    # plot aim dot
    ax1.plot(r_aim[0,0],r_aim[0,1],'*')


    #plot plates
    ax1.plot(rA2[0], rA2[1], '*', color='k')
    ax1.plot(A2_edges[0][[0,3],0],A2_edges[0][[0,3],1],  color='k', linewidth = 2)
    ax1.plot(A2_edges[1][[0,3],0],A2_edges[1][[0,3],1],  color='k', linewidth = 2)
    ax1.plot(rB2[0], rB2[1], '*', color='k')
    ax1.fill(B2_edges[0][:,0], B2_edges[0][:,1], fill=False, hatch='//', linewidth = 2)