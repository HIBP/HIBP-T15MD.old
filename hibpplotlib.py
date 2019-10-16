# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:01:34 2019

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import visvis as vv
import wire
import hibplib as hb
import pylab
import os
from scipy.stats import gaussian_kde


# %%
'''
   ############################################################################
   Functions for plotting Mafnetic field
   ############################################################################
'''
# %% visvis plot 3D
def plot_3d(B, wires, volume_corner1, volume_corner2, grid, resolution, cutoff=2):
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

    vv.volshow2(vol, renderStyle='mip', cm=vv.CM_JET)
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
    pf_coils = hb.importPFCoils('PFCoils.dat')
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
#        ax.quiver(points[:, 0], points[:, 1], B[:, 0], B[:, 1], scale=20.0)

        X = np.unique(points[:, 0])
        Y = np.unique(points[:, 1])
        cs = ax.contour(X, Y, Babs.reshape([len(X), len(Y)]).T, n_contours)
        ax.clabel(cs)

        plt.xlabel('x')
        plt.ylabel('y')

    plt.axis('equal')

    clb = plt.colorbar(cs)
    clb.set_label('V', labelpad=-40, y=1.05, rotation=0)

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
'''
   ############################################################################
   Functions for plotting electric field
   ############################################################################
'''
# %%
def plot_contours(X, Y, Z, U, n_contours=30,
                  tick_width=2, axis_labelsize=18, title_labelsize=18):
    '''
    contour plot of potential U
    :param X, Y, Z: mesh ranges in X, Y and Z respectively [m]
    :param U:  plate's U  [V]
    :param n_contours:  number of planes to skip before plotting
    :return: None
    '''
#    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig, ax1 = plt.subplots()

    # Grids
    ax1.grid(True)
    ax1.grid(which='major', color = 'tab:gray') #draw primary grid
    ax1.minorticks_on() # make secondary ticks on axes
    ax1.grid(which='minor', color = 'tab:gray', linestyle = ':') # draw secondary grid

#    ax2.grid(True)
#    ax2.grid(which='major', color = 'tab:gray') #draw primary grid
#    ax2.minorticks_on() # make secondary ticks on axes
#    ax2.grid(which='minor', color = 'tab:gray', linestyle = ':') # draw secondary grid

    # Axis

    ax1.xaxis.set_tick_params(width=tick_width) # increase tick size
    ax1.yaxis.set_tick_params(width=tick_width)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.axis('equal')

#    ax2.xaxis.set_tick_params(width=tick_width) # increase tick size
#    ax2.yaxis.set_tick_params(width=tick_width)
#    ax2.set_xlabel('Z (m)')
#    ax2.set_ylabel('Y (m)')
#    ax2.axis('equal')

    CS = ax1.contourf(X, Y, U[:,:,U.shape[2]//2].swapaxes(0, 1), n_contours)

    # add the edge of the domain
    domain = patches.Rectangle((min(X), min(Y)), max(X)-min(X), max(Y)-min(Y),
                               linewidth=2, linestyle='--', edgecolor='k',
                               facecolor='none')
    ax1.add_patch(domain)

#    ax2.contour(Z, Y, U[U.shape[0]//2, :, :], n_contours)
#
#    # add the edge of the domain
#    domain = patches.Rectangle((min(Z), min(Y)), max(Z)-min(Z), max(Y)-min(Y),
#                               linewidth=2, linestyle='--', edgecolor='k',
#                               facecolor='none')
#    ax2.add_patch(domain)

    ax1.set(xlim=(-0.12, 0.12), ylim=(-0.12, 0.12), autoscale_on=False)

    clb = plt.colorbar(CS)
    clb.set_label('V', labelpad=-40, y=1.05, rotation=0)

# %%
def plot_quiver(X, Y, Z, Ex, Ey, Ez, A2_edges):
    '''
    quiver plot of Electric field in xy, xz, zy planes
    :param X, Y, Z: mesh ranges in X, Y and Z respectively [m]
    :param Ex, Ey, Ez:  plate's U gradient components [V/m]
    :param n_skip:  number of planes to skip before plotting
    :return: None
    '''
#    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    fig, ax1 = plt.subplots(nrows=1, ncols=1)

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


#    x_cut = Ey.shape[0]//2 # x position of ZY cut
#    ax2.quiver(Z, Y, Ez[x_cut, :, :],
#                     Ey[x_cut, :, :])
#    ax2.set_xlabel('Z (m)')
#    ax2.set_ylabel('Y (m)')
#    ax2.grid(True)
#    ax2.axis('equal')
#    # add the edge of the domain
#    domain = patches.Rectangle((min(Z), min(Y)), max(Z)-min(Z), max(Y)-min(Y),
#                               linewidth=2, linestyle='--', edgecolor='k',
#                               facecolor='none')
#    ax2.add_patch(domain)
#
#    y_cut = Ex.shape[1]//2 # y position of XZ cut
#    ax3.quiver(X, Z, Ex[:, y_cut, :].swapaxes(0, 1),
#                     Ez[:, y_cut, :].swapaxes(0, 1))
#    ax3.set_xlabel('X (m)')
#    ax3.set_ylabel('Z (m)')
#    ax3.grid(True)
#    ax3.axis('equal')
#    # add the edge of the domain
#    domain = patches.Rectangle((min(X), min(Z)), max(X)-min(X), max(Z)-min(Z),
#                               linewidth=2, linestyle='--', edgecolor='k',
#                               facecolor='none')
#    ax3.add_patch(domain)

# %%
def plot_quiver3d(X, Y, Z, Ex, Ey, Ez, UP_rotated, LP_rotated, n_skip=5):
    '''
    3d quiver plot of Electric field
    :param X, Y, Z: mesh ranges in X, Y and Z respectively
    :param Ex, Ey, Ez:  plate's U gradient components
    :param UP_rotated, LP_rotated: upper's and lower's plate angle coordinates
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
#    z_pos = Z.shape[2]//2

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
'''
   ############################################################################
   Functions for plotting trajectories
   ############################################################################
'''

def plot_geometry(ax, coildata_filename = 'coildata.dat',
                      camera_data_filename = 'T15_vessel.txt',
                      separatrix_data_filename = 'T15_sep.txt',
                      PFCoils_data_filename = 'PFCoils.dat',
                      major_radius = 1.5):
    '''
    plot toroidal and poloidal field coils, camera and separatrix
    
    :param ax:                      graph to plot geometry on
    
    :param coil_data_filename:      txt or dat file with inner 
                                    and outer coil profile coords
                                 
    :param camera_data_filename:    txt or dat file with camera coords
    
    :separatrix_data_filename:      txt or dat file with plasma's 
                                    separatrix coords
    
    :PFCoils_data_filename:         txt or dat file with poloidal field 
                                    coils coords
    
    :major_radius:                  major radius of fusion reactor,
                                    used for moving separatrix and camera
    :return: None
    '''
    # load coil contours from txt
    array = np.loadtxt(coildata_filename)  # [m]
    
    # array has only x and y columns, so soon we need to add a zero column for z
    # (because we have no z column in coildata)
    outer_coil = np.array(array[:, [2, 3]])
    inner_coil = np.array(array[:, [0, 1]])
    # plot toroidal coil
    ax.plot(inner_coil[:, 0], inner_coil[:, 1], '--', color='k')
    ax.plot(outer_coil[:, 0], outer_coil[:, 1], '--', color='k')

    # get T-15 camera and plasma contours
    camera = np.loadtxt(camera_data_filename)/1000
    ax.plot(camera[:, 0] + major_radius, camera[:, 1], color='tab:blue')

    if separatrix_data_filename is not None:
        separatrix = np.loadtxt(separatrix_data_filename)/1000
        ax.plot(separatrix[:, 0] + major_radius, separatrix[:, 1], color='tab:orange')
    
    if PFCoils_data_filename is not None:
        pf_coils = hb.importPFCoils(PFCoils_data_filename)

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
def plot_traj_xy(traj_list, r_aim, A2_edges, B2_edges,
             Ebeam, UA2, Btor, Ipl,
             tick_width=2, font_size=24, title_fontsize=18, 
             axis_enabled = True):
    '''
    plot fan of trajectories in xy
    :param traj_list: list of trajectories
    :param r_aim: aim dot coordinates [m]
    :param A2_edges: A2 edges coordinates [m]
    :param B2_edges: B2 edges coordinates [m]
    :param Ebeam: beam energy [keV]
    :param Btor: toroidal magnetic field [T]
    :param Ipl: plasma current [MA]
    :return: None
    '''
    fig, ax1 = plt.subplots()

    # Grids
    ax1.grid(True)
    ax1.grid(which='major', color = 'tab:gray') #draw primary grid
    ax1.minorticks_on() # make secondary ticks on axes
    ax1.grid(which='minor', color = 'tab:gray', linestyle = ':') # draw secondary grid

    # Fonts and ticks
    plt.tick_params(axis='both', which='major', labelsize=font_size) # increase label font size
    ax1.xaxis.set_tick_params(width=2) # increase tick size
    ax1.yaxis.set_tick_params(width=2)
    ax1.set_xlabel('X (m)', fontsize=font_size)
    ax1.set_ylabel('Y (m)', fontsize=font_size)

    # plot axis
    if axis_enabled:
        axis_linewidth = 1.5
        axis_color = 'k'
        axis_linestyle = '--'
        ax1.plot([1.5,1.5],[-3,2.4], 
                 linestyle=axis_linestyle, color=axis_color, 
                 linewidth=axis_linewidth)
        ax1.plot([0.2,3.4],[0.,0.],
                 linestyle=axis_linestyle, color=axis_color, 
                 linewidth=axis_linewidth)

    # get T-15 camera and plasma contours
    plot_geometry(ax1)

    # plot aim dot
#    ax1.plot(r_aim[0,0],r_aim[0,1],'*')


#    for tr in traj_list:
#        if tr.Ebeam == Ebeam and tr.UA2 == UA2:
#            for i in tr.Fan:
#                ax1.plot(i[:,0], i[:,1],color='r')
#            #plot plates
#            ax1.plot(A2_edges[0][[0,3],0],A2_edges[0][[0,3],1],  color='k', linewidth = 2)
#            ax1.plot(A2_edges[1][[0,3],0],A2_edges[1][[0,3],1],  color='k', linewidth = 2)
#            ax1.fill(B2_edges[0][:,0], B2_edges[0][:,1], fill=False, hatch='//', linewidth = 2)
#            ax1.plot(tr.RV_Prim[:,0], tr.RV_Prim[:,1], color='k',  linewidth = 2)

    #plot plates
    ax1.plot(A2_edges[0][[0,3],0],A2_edges[0][[0,3],1],  color='k', linewidth = 2)
    ax1.plot(A2_edges[1][[0,3],0],A2_edges[1][[0,3],1],  color='k', linewidth = 2)
    ax1.fill(B2_edges[0][:,0], B2_edges[0][:,1], fill=False, hatch='//', linewidth = 2)


    for tr in traj_list:
        if tr.Ebeam == Ebeam and tr.UA2 == UA2:

#            ax1.plot(tr.RV_Sec[:,0], tr.RV_Sec[:,1],color='r')
#            index = np.where(np.round(tr.RV_Prim[:,0],3) == np.round(tr.RV_Sec[0,0],3))[0][0]
            index = np.where(np.round(tr.RV_Prim[:,0],3) == np.round(tr.Fan[15][0,0],3))[0][0] + 1
            ax1.plot(tr.RV_Prim[:,0][:index],
                     tr.RV_Prim[:,1][:index],
                     color='k',  linewidth = 2)

            ax1.plot(tr.Fan[15][:,0], tr.Fan[15][:,1],color='r')

#    ax1.set_title('E={} keV, UA2={} kV, Btor = {} T, Ipl = {} MA'.format(tr.Ebeam,tr.UA2, Btor, Ipl), fontsize=20)

#    # these are matplotlib.patch.Patch properties
#    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#    # place a text box in upper left in axes coords
#    ax1.text(0.5, 0.5, textstr, transform=ax1.transAxes, fontsize=18,
#            verticalalignment='top', bbox=props)

    ax1.set(xlim=(0.9, 4.28), ylim=(-1, 1.5), autoscale_on=False)

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
    :param Ipl: plasma current [MA]
    :return: None
    '''

#    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
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

#    ax3.grid(True)
#    ax3.grid(which='major', color = 'tab:gray') #draw primary grid
#    ax3.minorticks_on() # make secondary ticks on axes
#    ax3.grid(which='minor', color = 'tab:gray', linestyle = ':') # draw secondary grid

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

#    ax3.xaxis.set_tick_params(width=tick_width) # increase tick size
#    ax3.yaxis.set_tick_params(width=tick_width)
#    ax3.set_xlabel('Z (m)')
#    ax3.set_ylabel('Y (m)')
#    ax3.axis('equal')

    # get T-15 camera and plasma contours
    plot_geometry(ax1)

    # plot aim dot
    ax1.plot(r_aim[0,0],r_aim[0,1],'*')
    ax2.plot(r_aim[0,0],r_aim[0,2],'*')
#    ax3.plot(r_aim[0,2],r_aim[0,1],'*')

    for tr in traj_list:
        if tr.Ebeam == Ebeam and tr.UA2 == UA2:
            #plot plates
            ax1.plot(A2_edges[0][[0,3],0],A2_edges[0][[0,3],1],
                     color='k', linewidth = 2)
            ax1.plot(A2_edges[1][[0,3],0],A2_edges[1][[0,3],1],
                     color='k', linewidth = 2)
            ax1.fill(B2_edges[0][:,0], B2_edges[0][:,1], fill=False,
                     hatch='//', linewidth = 2)

            ax2.plot(B2_edges[0][[0,3],0],B2_edges[0][[0,3],2],
                     color='k', linewidth = 2)
            ax2.plot(B2_edges[1][[0,3],0],B2_edges[1][[0,3],2],
                     color='k', linewidth = 2)
            ax2.fill(A2_edges[0][:,0], A2_edges[0][:,2], fill=False,
                     hatch='//', linewidth = 2)

            ax1.plot(tr.RV_Prim[:,0], tr.RV_Prim[:,1],color='k')
            ax2.plot(tr.RV_Prim[:,0], tr.RV_Prim[:,2],color='k')
#            ax3.plot(tr.RV_Prim[:,2], tr.RV_Prim[:,1],color='k')

            last_points = []
            for i in tr.Fan:
                ax1.plot(i[:,0], i[:,1],color='r')
                ax2.plot(i[:,0], i[:,2],color='r')
#                ax3.plot(i[:,2], i[:,1],color='r')
                last_points.append(i[-1, :])
            last_points = np.array(last_points)
#            ax3.plot(last_points[:, 2], last_points[:, 1], '--o', color='r')

    ax1.set_title('E={} keV, UA2={} kV, UB2={} kV, Btor={} T, Ipl={} MA'.format(tr.Ebeam,tr.UA2, tr.UB2, Btor, Ipl))
#%%
def plot_fan_xy(traj_list, r_aim, A2_edges, B2_edges,
             Ebeam, UA2, Btor, Ipl,
             tick_width=2, font_size=24, title_fontsize=18, 
             axis_enabled = True):
    '''
    plot fan of trajectories in xy
    :param traj_list: list of trajectories
    :param r_aim: aim dot coordinates [m]
    :param A2_edges: A2 edges coordinates [m]
    :param B2_edges: B2 edges coordinates [m]
    :param Ebeam: beam energy [keV]
    :param Btor: toroidal magnetic field [T]
    :param Ipl: plasma current [MA]
    :return: None
    '''
    
    figure_name = 'Fan_E{}_U{}_alpha{}_beta{}_aim{}_{}_{}'\
                           .format(Ebeam, UA2, 
                                   traj_list[-1].alpha, traj_list[-1].beta,
                                   r_aim[0,0], r_aim[0,1], r_aim[0,2])

    fig, ax1 = plt.subplots(figsize=(8,7.5), num=figure_name)

    # Grids
    ax1.grid(True)
    ax1.grid(which='major', color = 'tab:gray') #draw primary grid
    ax1.minorticks_on() # make secondary ticks on axes
    ax1.grid(which='minor', color = 'tab:gray', linestyle = ':') # draw secondary grid

    # Fonts and ticks
    plt.tick_params(axis='both', which='major', labelsize=font_size) # increase label font size
    ax1.xaxis.set_tick_params(width=2) # increase tick size
    ax1.yaxis.set_tick_params(width=2)
    ax1.set_xlabel('X (m)', fontsize=font_size)
    ax1.set_ylabel('Y (m)', fontsize=font_size)

    # plot axis
    if axis_enabled:
        axis_linewidth = 1.5
        axis_color = 'k'
        axis_linestyle = '--'
        ax1.plot([1.5,1.5],[-3,2.4], 
                 linestyle=axis_linestyle, color=axis_color, 
                 linewidth=axis_linewidth)
        ax1.plot([0.2,3.4],[0.,0.],
                 linestyle=axis_linestyle, color=axis_color, 
                 linewidth=axis_linewidth)

    # get T-15 camera and plasma contours
    plot_geometry(ax1)

    #plot plates
    ax1.plot(A2_edges[0][[0,3],0],A2_edges[0][[0,3],1],  color='k', linewidth = 2)
    ax1.plot(A2_edges[1][[0,3],0],A2_edges[1][[0,3],1],  color='k', linewidth = 2)
    ax1.fill(B2_edges[0][:,0], B2_edges[0][:,1], fill=False, hatch='//', linewidth = 2)


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

            last_points = []
            for idx in range(len(tr.Fan)):
                if idx % 2:
                    i = tr.Fan[idx]
                    ax1.plot(i[:,0], i[:,1],color='r')
                    last_points.append(i[-1, :])
            last_points = np.array(last_points)

    # plot aim dot
    ax1.plot(r_aim[0,0],r_aim[0,1],'*')

    ax1.set(xlim=(1.25, 3.5), ylim=(-0.75, 1.5), autoscale_on=False)

# %%
def plot_scan(traj_list, r_aim,A2_edges, B2_edges,
              Ebeam, Btor, Ipl,
              tick_width=2, axis_labelsize=18, title_labelsize=18):

    '''
    plot scan for one beam with particular energy in 2 planes: xy, xz
    :param traj_list: list of trajectories
    :param r_aim: aim dot coordinates [m]
    :param Ebeam: beam energy [keV]
    :param Btor: toroidal magnetic field [T]
    :param Ipl: plasma current [MA]
    :return: None
    '''
    figure_name = 'Scan_E{}_alpha{}_beta{}_aim{}_{}_{}'\
                           .format(Ebeam, traj_list[-1].alpha, traj_list[-1].beta,
                                   r_aim[0,0], r_aim[0,1], r_aim[0,2])

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, num=figure_name)

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
        if traj_list[i].Ebeam == Ebeam:
            A2list.append(traj_list[i].UA2)

    #find UA2 max and min
    UA2_max = np.amax(np.array(A2list))
    UA2_min = np.amin(np.array(A2list))

    ax1.set_title('Ebeam={} keV, UA2:[{}, {}] kV, Btor = {} T, Ipl = {} MA'
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
def plot_scan_xy(traj_list, r_aim, A2_edges, B2_edges,
              Ebeam, Btor, Ipl,
              tick_width=2, font_size=24, title_labelsize=18,
              axis_enabled = True,
              plot_ionization_line = True):

    '''
    plot scan for one beam with particular energy in 2 planes: xy, xz
    :param traj_list: list of trajectories
    :param r_aim: aim dot coordinates [m]
    :param Ebeam: beam energy [keV]
    :param Btor: toroidal magnetic field [T]
    :param Ipl: plasma current [MA]
    :param RV_sec_first: array containing first points of secondary trajectories
    :return: None
    '''
    figure_name = 'Scan_E{}_alpha{}_beta{}_aim{}_{}_{}'\
                           .format(Ebeam, traj_list[-1].alpha, traj_list[-1].beta,
                                   r_aim[0,0], r_aim[0,1], r_aim[0,2])

    fig, ax1 = plt.subplots(figsize=(8,7.5), num=figure_name)

    # Grids
    ax1.grid(True)
    ax1.grid(which='major', color = 'tab:gray') #draw primary grid
    ax1.minorticks_on() # make secondary ticks on axes
    ax1.grid(which='minor', color = 'tab:gray', linestyle = ':') # draw secondary grid

    # Fonts and ticks
    plt.tick_params(axis='both', which='major', labelsize=font_size) # increase label font size
    ax1.xaxis.set_tick_params(width=2) # increase tick size
    ax1.yaxis.set_tick_params(width=2)
    ax1.set_xlabel('X (m)', fontsize=font_size)
    ax1.set_ylabel('Y (m)', fontsize=font_size)

    # plot axis
    if axis_enabled:
        axis_linewidth = 1.5
        axis_color = 'k'
        axis_linestyle = '--'
        ax1.plot([1.5,1.5],[-3,2.4], 
                 linestyle=axis_linestyle, color=axis_color, 
                 linewidth=axis_linewidth)
        ax1.plot([0.2,3.4],[0.,0.],
                 linestyle=axis_linestyle, color=axis_color, 
                 linewidth=axis_linewidth)

    # get T-15 camera and plasma contours
    plot_geometry(ax1)
    
    # plot aim dot
    ax1.plot(r_aim[0,0],r_aim[0,1],'*')

    # get the list of UA2
    A2list = []
    for i in range(len(traj_list)):
        if traj_list[i].Ebeam == Ebeam:
            A2list.append(traj_list[i].UA2)
            try:
                RV_sec_first = np.vstack((RV_sec_first, 
                                          traj_list[i].RV_Sec[0,:2]))
            except UnboundLocalError:
                RV_sec_first = traj_list[i].RV_Sec[0,:2]

#    #find UA2 max and min
    UA2_max = np.amax(np.array(A2list))
    UA2_min = np.amin(np.array(A2list))

    ax1.set_title('UA2:[{}, {}] kV'.format(UA2_min,  UA2_max), 
                  fontsize = title_labelsize)
                
    ax1.plot(A2_edges[0][[0,3],0],A2_edges[0][[0,3],1],  color='k', linewidth = 2)
    ax1.plot(A2_edges[1][[0,3],0],A2_edges[1][[0,3],1],  color='k', linewidth = 2)
    ax1.fill(B2_edges[0][:,0], B2_edges[0][:,1], fill=False, hatch='//', linewidth = 2)

    for i in range(len(traj_list)):
        if traj_list[i].Ebeam == Ebeam:
            # plot trajectories
            ax1.plot(traj_list[i].RV_Prim[:,0], traj_list[i].RV_Prim[:,1], color='k')
            ax1.plot(traj_list[i].RV_Sec[:,0], traj_list[i].RV_Sec[:,1], color='r')

    if plot_ionization_line:
        ax1.plot(RV_sec_first[:,0],RV_sec_first[:,1],'o', color='darkred', linestyle='--' )

    ax1.set(xlim=(1.25, 3.5), ylim=(-0.75, 1.5), autoscale_on=False)
#    ax1.set(xlim=(1.0, 2.5), ylim=(-0.5, 1.1), autoscale_on=False)
    ax1.tick_params(axis='both', which='major', labelsize=16)

#%%
def plot_scan_xz(traj_list, r_aim, A2_edges, B2_edges,
              Ebeam, Btor, Ipl,
              tick_width=2, axis_labelsize=18, title_labelsize=18):

    '''
    plot scan for one beam with particular energy in 2 planes: xy, xz
    :param traj_list: list of trajectories
    :param r_aim: aim dot coordinates [m]
    :param Ebeam: beam energy [keV]
    :param Btor: toroidal magnetic field [T]
    :param Ipl: plasma current [MA]
    :param RV_sec_first: array containing first points of secondary trajectories
    :return: None
    '''

    figure_name = 'Scan_xz_E{}_alpha{}_beta{}_aim{}_{}_{}'\
                           .format(int(Ebeam), traj_list[-1].alpha, traj_list[-1].beta,
                                   r_aim[0,0], r_aim[0,1], r_aim[0,2])
                           
    fig, ax1 = plt.subplots(figsize=(8,7.5), num=figure_name)

    # Grids
    ax1.grid(True)
    ax1.grid(which='major', color = 'tab:gray') #draw primary grid
    ax1.minorticks_on() # make secondary ticks on axes
    ax1.grid(which='minor', color = 'tab:gray', linestyle = ':') # draw secondary grid

    # Axis
    ax1.xaxis.set_tick_params(width=tick_width) # increase tick size
    ax1.yaxis.set_tick_params(width=tick_width)
    ax1.set_ylabel('Z (m)', fontsize=16)
    ax1.set_xlabel('X (m)', fontsize=16)
    
    # plot geometry
    ax1.add_patch(plt.Circle((0,0), radius=2.4, fill=False,color = 'tab:blue'))
    ax1.add_patch(plt.Circle((0,0), radius=2.1, fill=False,color = 'tab:orange'))

    # get the list of UA2
    B2list = []
    for i in range(len(traj_list)):
        if traj_list[i].Ebeam == Ebeam:
            B2list.append(traj_list[i].UB2)
            try:
                RV_sec_first = np.vstack((RV_sec_first, traj_list[i].RV_Sec[0,[0,2]]))
            except UnboundLocalError:
                RV_sec_first = traj_list[i].RV_Sec[0,[0,2]]

#    #find UB2 max and min
    UB2_max = np.amax(np.array(B2list))
    UB2_min = np.amin(np.array(B2list))

    ax1.set_title('UB2:[{}, {}] kV'.format(round(UB2_min,1), round(UB2_max,1)),
                  fontsize=title_labelsize)

    for i in range(len(traj_list)):
        if traj_list[i].Ebeam == Ebeam:
            # plot trajectories
            ax1.plot(traj_list[i].RV_Prim[:,0], traj_list[i].RV_Prim[:,2], color='k')

    for i in range(len(traj_list)):
        if traj_list[i].Ebeam == Ebeam:
            ax1.plot(traj_list[i].RV_Sec[:,0], traj_list[i].RV_Sec[:,2], color='r')
            
    ax1.plot(RV_sec_first[:,0],RV_sec_first[:,1],'o', color='darkred', linestyle='--' )

    # plot plates
    ax1.plot(B2_edges[0][[0,3],0],B2_edges[0][[0,3],2],
             color='k', linewidth = 2)
    ax1.plot(B2_edges[1][[0,3],0],B2_edges[1][[0,3],2],
             color='k', linewidth = 2)
    ax1.fill(A2_edges[0][:,0], A2_edges[0][:,2], fill=False,
             hatch='//', linewidth = 2)
    
    ax1.set(xlim=(1.25, 2.75), ylim=(-0.75, 0.75), autoscale_on=False)
    ax1.tick_params(axis='both', which='major', labelsize=16)

    # plot aim dot
    ax1.plot(r_aim[0,0],r_aim[0,2],'*')

#%%
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
    :param Ipl: plasma current [MA]
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

#    # find UA2 max and min
#    UA2_max = np.amax(np.array(A2list))
#    UA2_min = np.amin(np.array(A2list))

#    ax1.set_title('Ebeam:[{}, {}] keV, UA2:[{}, {}] kV, Btor = {} T, Ipl = {} MA'
#                  .format(traj_list[0].Ebeam, traj_list[-1].Ebeam,
#                         UA2_min,  UA2_max,
#                          Btor, Ipl))

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

#    ax1.set(xlim=(0.9, 4.28), ylim=(-1, 1.5), autoscale_on=False)

#%%
def plot_grid_xy(traj_list, r_aim, Btor, Ipl, 
                 legend=False, zoom=True, axis_enabled=True,
                 font_size = 24,
                 linestyle_A2='--', linestyle_E='-', 
                 marker_A2='*', marker_E='p',
                 traj_color='tab:gray'):
    '''
    plot detector grid in 2 planes: xy, xz
    :param traj_list: list of trajectories
    :param r_aim: aim dot coordinates [m]
    :param Btor: toroidal magnetic field [T]
    :param Ipl: plasma current [MA]
    :return: grid figure
    '''
    # Create a name for a figure, 
    # that consists of injection angles and aim dot coords
    figure_name = 'Grid_alpha{}_beta{}_aim{}_{}_{}'\
                           .format(traj_list[-1].alpha, traj_list[-1].beta,
                                   r_aim[0,0], r_aim[0,1], r_aim[0,2])
                           
    fig, ax1 = plt.subplots(figsize=(8,7.5), num=figure_name)
#    # Grids
    ax1.grid(True)
    ax1.grid(which='major', color = 'tab:gray') #draw primary grid
    ax1.minorticks_on() # make secondary ticks on axes
    ax1.grid(which='minor', color = 'tab:gray', linestyle = ':') # draw secondary grid
#
#    # Fonts and ticks
    plt.tick_params(axis='both', which='major', labelsize=font_size) # increase label font size
    ax1.xaxis.set_tick_params(width=2) # increase tick size
    ax1.yaxis.set_tick_params(width=2)
    ax1.set_xlabel('X (m)', fontsize=font_size)
    ax1.set_ylabel('Y (m)', fontsize=font_size)

#    #get T-15 camera and plasma contours
    plot_geometry(ax1)

#  # plot axis
    if axis_enabled:
        axis_linewidth = 1.5
        axis_color = 'k'
        axis_linestyle = '--'
        ax1.plot([1.5,1.5],[-3,2.4], 
                 linestyle=axis_linestyle, color=axis_color, 
                 linewidth=axis_linewidth)
        ax1.plot([0.2,3.4],[0.,0.],
                 linestyle=axis_linestyle, color=axis_color, 
                 linewidth=axis_linewidth)

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

#    #find UA2 max and min
#    UA2_max = np.amax(np.array(A2list))
#    UA2_min = np.amin(np.array(A2list))

#    ax1.set_title('Ebeam:[{}, {}] keV, UA2:[{}, {}] kV, Btor = {} T, Ipl = {} MA'
#                  .format(traj_list[0].Ebeam, traj_list[-1].Ebeam,
#                         UA2_min,  UA2_max,
#                          Btor, Ipl))

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


#    ax1.plot(r_aim[0,0],r_aim[0,1],'*')


    
    if legend:
        ax1.legend(title='Tl', fontsize = font_size, title_fontsize = font_size+8)
    if zoom:
        ax1.set(xlim=(1.3, 2.5), ylim=(-0.4, 1.1))
        
        #xvalues = np.arange(1.3, 2.5, 0.5)
        #ax1.set_xticks(xvalues)
        
#        ax1.set(xlim=(0, 3), ylim=(-0.4, 2), autoscale_on=False)
        

    plt.show()
    
    return fig, ax1, figure_name

#%%
def plot_grid_xz(traj_list, r_aim, Btor, Ipl, legend=False, zoom=True,
                 linestyle_A2='--', linestyle_E='-', 
                 marker_A2='*', marker_E='p',
                 traj_color='tab:gray'):
    '''
    plot detector grid in 2 planes: xy, xz
    :param traj_list: list of trajectories
    :param r_aim: aim dot coordinates [m]
    :param Btor: toroidal magnetic field [T]
    :param Ipl: plasma current [MA]
    :return: grid figure
    '''
    # Create a name for a figure, 
    # that consists of injection angles and aim dot coords
    figure_name = 'Grid_xz_alpha{}_beta{}_aim{}_{}_{}'\
                           .format(traj_list[-1].alpha, traj_list[-1].beta,
                                   r_aim[0,0], r_aim[0,1], r_aim[0,2])
                           
    fig, ax1 = plt.subplots(figsize=(8,7.5), num=figure_name)
    
#    # Grids
    ax1.grid(True)
    ax1.grid(which='major', color = 'tab:gray') #draw primary grid
    ax1.minorticks_on() # make secondary ticks on axes
    ax1.grid(which='minor', color = 'tab:gray', linestyle = ':') # draw secondary grid
#
#    # Fonts and ticks
    plt.tick_params(axis='both', which='major', labelsize=16) # increase label font size
    ax1.xaxis.set_tick_params(width=2) # increase tick size
    ax1.yaxis.set_tick_params(width=2)
    ax1.set_xlabel('X (m)', fontsize=16)
    ax1.set_ylabel('Z (m)', fontsize=16)

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

    # make a grid of constant A2
    for i_A2 in range(0,N_A2,1):
        k = -1
        for i_tr in range(len(traj_list)):
            if traj_list[i_tr].UA2 == A2list[i_A2]:
                k += 1
                # take the 1-st point of secondary trajectory
                x = traj_list[i_tr].RV_Sec[0,0]
                y = traj_list[i_tr].RV_Sec[0,1]
                z = traj_list[i_tr].RV_Sec[0,2]
                A2_grid[k,:,i_A2] = [x,y,z]


        ax1.plot(A2_grid[:,0,i_A2], A2_grid[:,2,i_A2],
                 linestyle=linestyle_A2,
                 marker=marker_A2,
                 label=str(round(A2list[i_A2],1))+' kV')

    # make a grid of constant E
    for i_E in range(0,N_E,1):
        k = -1
        for i_tr in range(len(traj_list)):
            if traj_list[i_tr].Ebeam == Elist[i_E]:
                k += 1
                # take the 1-st point of secondary trajectory
                x = traj_list[i_tr].RV_Sec[0,0]
                y = traj_list[i_tr].RV_Sec[0,1]
                z = traj_list[i_tr].RV_Sec[0,2]
                E_grid[k,:,i_E] = [x,y,z]

        ax1.plot(E_grid[:,0,i_E], E_grid[:,2,i_E],
                 linestyle=linestyle_E,
                 marker=marker_E,
                 label=str(round(Elist[i_E],1))+' keV')


    # plot raim
#    ax1.plot(r_aim[0,0],r_aim[0,2], markerfacecolor='xkcd:black', 
 #            markersize=23, marker='<')


    if legend:
        ax1.legend(title='Tl', fontsize = 16, title_fontsize = 24)
    if zoom:
        ax1.set(xlim=(1.0, 2.5), ylim=(-1.25, 1.25), autoscale_on=False)

    plt.show()
    
    return fig, ax1, figure_name
    
#%%
    
def plot_legend(ax, figure_name):
    """
    plots legendin separate window
    
    Args:
    :ax - axes to get legnd from
    :figure_name - get figure's name as base for legend's file name
    
    return figure object
    """
    # create separate figure for legend
    figlegend = plt.figure(num='Legend_for_' + figure_name, figsize=(1, 12))

    # get legend from ax
    figlegend.legend(*ax.get_legend_handles_labels(), loc="center")
    plt.show()

    return figlegend
    
        
#%%

def diplo_3Dnet():

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    l = 5
    for x in range(l):
        for y in range(l):
            for z in range(l):
                if (x == 0) or (y == 4) or (z == 0):
                    ax.scatter(x-l//2, y-l//2, z-l//2, c='lightgray', marker='o')
                if (x in [1,2,3]) and (y in [1,2,3]) and (z in [1,2,3]) \
                    and not ((x in [3]) and (y in [1]) and (z in [3])) \
                    and not ((x in [2,3]) and (y in [1]) and (z in [3])) \
                    and not ((x in [3]) and (y in [2]) and (z in [3])):
                    ax.scatter(x-l//2, y-l//2, z-l//2, c='k', marker='o')

    x = [-1,0,1]

    ax.plot(x,[0,0,0],[0,0,0] ,color = 'r', marker='o', linestyle = '--')
    ax.plot([0,0,0], x,[0,0,0] ,color = 'r', marker='o', linestyle = '--')
    ax.plot([0,0,0], [0,0,0], x ,color = 'r', marker='o', linestyle = '--')
        
                    
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()
    
#%%
def PlotAngles1En(traj_list, Btor, Ipl, Ebeam):
    equal_energy_list = []
    for i in range(len(traj_list)):
        if traj_list[i].Ebeam == Ebeam:
            # Get vector coords for last point in Secondary traj
            Vx = traj_list[i].RV_Sec[-1,3] # Vx
            Vy = traj_list[i].RV_Sec[-1,4] # Vy
            Vz = traj_list[i].RV_Sec[-1,5] # Vz
            equal_energy_list.append([traj_list[i].UA2, traj_list[i].UB2,
                                      np.arctan(Vy/np.sqrt(Vx**2 + Vy**2))*180/np.pi,
                                      np.arctan(-Vz/Vx)*180/np.pi])

    equal_energy_list = np.array(equal_energy_list)

    #find  max and min angles
    angles_min = np.minimum.reduce(equal_energy_list)
    angles_max = np.maximum.reduce(equal_energy_list)


    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    ax1.plot(equal_energy_list[:,0], equal_energy_list[:,2])
    ax1.axis('tight')
    ax1.set_xlabel('UA2 (kV)')
    ax1.set_ylabel('Exit alpha (grad)')

    ax2.plot(equal_energy_list[:,1], equal_energy_list[:,3])
    ax2.axis('tight')
    ax2.set_xlabel('UB2 (kV)')
    ax2.set_ylabel('Exit beta (grad)')

    # Grids
    ax1.grid(True)
    ax1.grid(which='major', color = 'tab:gray') #draw primary grid
    ax1.minorticks_on() # make secondary ticks on axes
    ax1.grid(which='minor', color = 'tab:gray', linestyle = ':') # draw secondary grid

    # Fonts and ticks
    ax1.tick_params(axis='both', which='major', labelsize=18) # increase label font size
    ax1.xaxis.set_tick_params(width=2) # increase tick size
    ax1.yaxis.set_tick_params(width=2)

    ax1.set_title('exit alpha(UA2) for Ebeam: {} keV'.format(Ebeam))

    # Grids
    ax2.grid(True)
    ax2.grid(which='major', color = 'tab:gray') #draw primary grid
    ax2.minorticks_on() # make secondary ticks on axes
    ax2.grid(which='minor', color = 'tab:gray', linestyle = ':') # draw secondary grid

    # Fonts and ticks
    ax2.tick_params(axis='both', which='major', labelsize=18) # increase label font size
    ax2.xaxis.set_tick_params(width=2) # increase tick size
    ax2.yaxis.set_tick_params(width=2)

    ax2.set_title('exit beta(UA2) for Ebeam: {} keV'.format(Ebeam))
    fig.suptitle('Exit angles(U) for Ebeam: {} keV'.format(Ebeam)+
                  '\nUA2: [{}, {}] kV, UB2: [{}, {}] kV'
                  .format(angles_min[0], angles_max[0],
                          round(angles_min[1],1), round(angles_max[1],1))+
                  '\nalpha: [{}, {}] grad, beta: [{}, {}] grad'
                  .format(round(angles_min[2],1), round(angles_max[2],1),
                          round(angles_min[3],1), round(angles_max[3],1))+
                  '\nBtor = {} T, Ipl = {} MA'.format(Btor, Ipl), fontsize=14)

#%%

def PlotSecAngles(traj_list, Btor, Ipl, traj_list_passed, Ebeam = 'all'):
    traj_list = traj_list_passed

    equal_energy_list = []
    grouped_energy_dict = {}

    for i in range(len(traj_list)):
        if traj_list[i].Ebeam != traj_list[i-1].Ebeam:
            equal_energy_list = []
        # Get vector coords for last point in Secondary traj
        Vx = traj_list[i].RV_Sec[-1,3] # Vx
        Vy = traj_list[i].RV_Sec[-1,4] # Vy
        Vz = traj_list[i].RV_Sec[-1,5] # Vz
        angle_list = [traj_list[i].UA2, traj_list[i].UB2,
                                  np.arctan(Vy/np.sqrt(Vx**2 + Vy**2))*180/np.pi,
                                  np.arctan(-Vz/Vx)*180/np.pi]
        equal_energy_list.append(angle_list)  # put angle arrays with
                                               # equal energy in the
                                               # same array

        grouped_energy_dict[traj_list[i].Ebeam] = np.array(equal_energy_list)

    x = []
    y = []
    for energy in grouped_energy_dict:
        x = np.hstack([x,grouped_energy_dict[energy][:,0]])
        y = np.hstack([y,grouped_energy_dict[energy][:,2]])

    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    fig, ax1 = plt.subplots()
    ax1.scatter(x, y, c=z, s=50, edgecolor='')
    ax1.set_xlabel('UA2 (kV)')
    ax1.set_ylabel('Exit alpha (grad)')
    # Grids
    ax1.grid(True)
    ax1.grid(which='major', color = 'tab:gray') #draw primary grid
    ax1.minorticks_on() # make secondary ticks on axes
    ax1.grid(which='minor', color = 'tab:gray', linestyle = ':') # draw secondary grid

    # Fonts and ticks
    ax1.tick_params(axis='both', which='major', labelsize=18) # increase label font size
    ax1.xaxis.set_tick_params(width=2) # increase tick size
    ax1.yaxis.set_tick_params(width=2)



#    ax1.scatter(A2list[:,0], sec_angles_list[:,0],
#                             s=20, c=colors[i], alpha=0.5,
#                             label=str(round(traj_list[i].Ebeam),1)+' keV')


    #    ax1.legend(dot_list,
    #               ('Low Outlier', 'LoLo', 'Lo', 'Average', 'Hi', 'HiHi', 'High Outlier'),
    #               scatterpoints=1,
    #               loc='lower left',
    #               ncol=3,
    #               fontsize=8)

#    ax1.title('Ebeam:[{}, {}] keV, UA2:[{}, {}] kV, Btor = {} T, Ipl = {} MA'
#                  .format(traj_list[0].Ebeam, traj_list[-1].Ebeam,
#                         UA2_min,  UA2_max,
#                          Btor, Ipl))
#    return dot_list

#    ax2.plot(A2list[:,1], sec_angles_list[:,1])
#    ax2.axis('tight')
#    ax2.set_xlabel('UB2 (kV)')
#    ax2.set_ylabel('Exit beta (grad)')

