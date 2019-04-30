# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 14:28:09 2019

@author: User
"""

import matplotlib.pyplot as plt
import numpy as np
import time
from hibplib import Rotate
import os
import errno
'''
Calculate electric potential and electric field between two plates
'''
# %%

def PlateCoords(plate_length, plate_thic, plate_width, gap,
                plts_alpha, plts_beta, plts_gamma):
    '''
    Geometry rotated in system based on central point between plates
    upper plate
    '''

    UP1 =  np.array([-plate_length/2., gap/2. + plate_thic, plate_width/2.])
    UP2 =  np.array([-plate_length/2., gap/2. + plate_thic, -plate_width/2.])
    UP3 =  np.array([plate_length/2., gap/2. + plate_thic, -plate_width/2.])
    UP4 =  np.array([plate_length/2., gap/2. + plate_thic, plate_width/2.])
    UP5 =  np.array([-plate_length/2., gap/2., plate_width/2.])
    UP6 =  np.array([-plate_length/2., gap/2., -plate_width/2.])
    UP7 =  np.array([plate_length/2., gap/2., -plate_width/2.])
    UP8 =  np.array([plate_length/2., gap/2., plate_width/2.])
    UP_full = np.array([UP1, UP2, UP3, UP4, UP5, UP6, UP7, UP8])
    UP_rotated = UP_full.copy()
    for UP in range(UP_full.shape[0]):
        UP_rotated[UP, :] = Rotate(UP_rotated[UP, :], axis=(1, 0, 0), deg=plts_gamma)
        UP_rotated[UP, :] = Rotate(UP_rotated[UP, :], axis=(0, 0, 1), deg=plts_alpha)
        UP_rotated[UP, :] = Rotate(UP_rotated[UP, :], axis=(0, 1, 0), deg=plts_beta)


    # lower plate
    LP1 =  np.array([-plate_length/2., -gap/2. - plate_thic, plate_width/2.])
    LP2 =  np.array([-plate_length/2., -gap/2. - plate_thic, -plate_width/2.])
    LP3 =  np.array([plate_length/2., -gap/2. - plate_thic, -plate_width/2.])
    LP4 =  np.array([plate_length/2., -gap/2. - plate_thic, plate_width/2.])
    LP5 =  np.array([-plate_length/2., -gap/2., plate_width/2.])
    LP6 =  np.array([-plate_length/2., -gap/2., -plate_width/2.])
    LP7 =  np.array([plate_length/2., -gap/2., -plate_width/2.])
    LP8 =  np.array([plate_length/2., -gap/2., plate_width/2.])
    LP_full = np.array([LP1, LP2, LP3, LP4, LP5, LP6, LP7, LP8])
    LP_rotated = LP_full.copy()
    for LP in range(LP_full.shape[0]):
        LP_rotated[LP, :] = Rotate(LP_rotated[LP, :], axis=(1, 0, 0), deg=plts_gamma)
        LP_rotated[LP, :] = Rotate(LP_rotated[LP, :], axis=(0, 0, 1), deg=plts_alpha)
        LP_rotated[LP, :] = Rotate(LP_rotated[LP, :], axis=(0, 1, 0), deg=plts_beta)

    return UP_rotated, LP_rotated

# %%
def PlateFlags(upper_cube, lower_cube,
               plts_alpha, plts_beta, plts_gamma):
    ''' create masks for coordinates
        which get into plates defined by upper and lower cubes
    '''
    upper_plate_flag = np.full_like(U, False, dtype=bool)
    lower_plate_flag = np.full_like(U, False, dtype=bool)
    for i in range(mesh_range_x.shape[0]):
        for j in range(mesh_range_y.shape[0]):
            for k in range(mesh_range_z.shape[0]):
                x = mesh_range_x[i]
                y = mesh_range_y[j]
                z = mesh_range_z[k]

                if (x > upper_cube[0, 0]) & (x < upper_cube[1, 0]) &\
                   (y > upper_cube[0, 1]) & (y < upper_cube[1, 1]) &\
                   (z > upper_cube[0, 2]) & (z < upper_cube[1, 2]):
                       r_rot = Rotate(Rotate(Rotate(np.array([x, y, z]), axis=(0, 1, 0), deg= -plts_beta), \
                                      axis=(0, 0, 1), deg= -plts_alpha), axis=(1, 0, 0), deg= -plts_gamma)
                       # define masks for upper and lower plates
                       upper_plate_flag[i,j,k] = (r_rot[0] > -plate_length/2.)&(r_rot[0] < plate_length/2.)& \
                                     (r_rot[2] > -plate_width/2.)&(r_rot[2] < plate_width/2.)&   \
                                     (r_rot[1] > gap/2.)&(r_rot[1] < gap/2. + plate_thic )

                elif (x > lower_cube[0, 0]) & (x < lower_cube[1, 0]) &\
                     (y > lower_cube[0, 1]) & (y < lower_cube[1, 1]) &\
                     (z > lower_cube[0, 2]) & (z < lower_cube[1, 2]):
                       r_rot = Rotate(Rotate(Rotate(np.array([x, y, z]), axis=(0, 1, 0), deg= -plts_beta), \
                                      axis=(0, 0, 1), deg= -plts_alpha), axis=(1, 0, 0), deg= -plts_gamma)
                       # define masks for upper and lower plates
                       lower_plate_flag[i,j,k] = (r_rot[0] > -plate_length/2.)&(r_rot[0] < plate_length/2.)& \
                                     (r_rot[2] > -plate_width/2.)&(r_rot[2] < plate_width/2.)&   \
                                     (r_rot[1] > -gap/2. - plate_thic)&(r_rot[1] < -gap/2. )

    return upper_plate_flag, lower_plate_flag

# %%
def InitConditions(U, Uupper_plate, Ulower_plate,
                   plates_mask, angle, edge_flag):
    # plates conditions
    for label in plates_mask:
        if label == angle:
            U[plates_mask[label][0]] = Uupper_plate
            U[plates_mask[label][1]] = Ulower_plate

        else:
            U[plates_mask[label][0]] = Ulower_plate
            U[plates_mask[label][1]] = Ulower_plate
    # boundary conditions
    U[edge_flag] = 0.0
    return U

# %%
def PDEstep(U, Uupper_plate, Ulower_plate, plates_mask, angle, edge_flag):
    '''
    PDE calculation at a single time step t
    '''
    # apply initial conditions at every time step
    U = InitConditions(U, Uupper_plate, Ulower_plate,
                       plates_mask, angle, edge_flag)

    U[1:-1, 1:-1, 1:-1] = (U[0:-2, 1:-1, 1:-1] + U[2:, 1:-1, 1:-1] + \
                           U[1:-1, 0:-2, 1:-1] + U[1:-1, 2:, 1:-1] + \
                           U[1:-1, 1:-1, 0:-2] + U[1:-1, 1:-1, 2:])/6.
    return U

# %%
def SaveElectricField(fname, Ex, Ey, Ez, plts_alpha, plts_beta, gamma, dirname='elecfield'):
    '''
    save Ex, Ey, Ez arrays to file
    '''

    dirname = dirname + '/octupol'

    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname, 0o700)
            print("Directory " , dirname ,  " created ")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    open(dirname + '/' + fname, 'w').close() # erases data from file before writing

    with open(dirname + '/' + fname, 'w') as myfile:
        myfile.write('{} {} {} {} # plate\'s length, thic, width and gap\n'.format(plate_length, plate_thic, plate_width, gap))
        myfile.write('{} {} {} # plate\'s alpha, beta and gamma angle\n'.format(plts_alpha, plts_beta, gamma))
        myfile.write('{} {} {} # number of dots (x,y,z)\n'.format(Ex.shape[0], Ex.shape[1], Ex.shape[2]))
        myfile.write('{} {} {} {} # border x, border y, border z, delta\n'.format(border_x, border_y, border_z, delta))
        for i in range(Ex.shape[0]):
            for j in range(Ex.shape[1]):
                for k in range(Ex.shape[2]):
                    myfile.write('{} {} {}\n'.format(Ex[i,j,k], Ey[i,j,k], Ez[i,j,k]))

    print('Electric field saved, ' + fname)

# %%
if __name__ == '__main__':
    # define plates geometry
    plate_length = 0.15 # along X [m]
    plate_width = 0.05  # along Z [m]
    plate_thic = 0.002 # [m]
    gap = 0.12 # distance between plates along Y [m]
    n_plates = 8 # number of plates (for octupol n_plates = 8)

    # define center position
    plts2_center = np.array([0., 0., 0.]) # plates center
    plts2_alpha = 30. # plates alpha
    plts2_beta = 0. # plates beta
    plts2_gamma = -0. # plates gamma
    plts2_angles = np.array([plts2_alpha, plts2_beta, plts2_gamma])

    # define voltages [Volts]
    Uupper_plate = 0.
    Ulower_plate = 1e3

    # Create mesh grid
    border_x = round(2*plate_length, 2) # length of the X-edge of the domain [m]
    border_z = round(2*plate_width, 2)
    border_y = round(3*gap, 2)
    delta = plate_thic/2 # space step

    mesh_range_x = np.arange(-border_x/2., border_x/2., delta)
    mesh_range_y = np.arange(-border_y/2., border_y/2., delta)
    mesh_range_z = np.arange(-border_z/2., border_z/2., delta)
    x, y, z = np.meshgrid(mesh_range_x, mesh_range_y,
                          mesh_range_z, indexing='ij') # [X ,Y, Z]

    mx = mesh_range_x.shape[0]
    my = mesh_range_y.shape[0]
    mz = mesh_range_z.shape[0]

    # define mask for edge elements
    edge_flag = (x < -0.9*border_x/2.) | (x > 0.9*border_x/2.) | \
                (y < -0.9*border_y/2.) | (y > 0.9*border_y/2.) | \
                (z < -0.9*border_z/2.) | (z > 0.9*border_z/2.)

    # array for electric potential
    U = np.zeros((mx, my, mz))

    U0 = np.copy(U)
    U1 = np.full_like(U,1e3)




    # Stack both plates arrays into one dict
    plates_rotated = {}
    plates_mask = {}
    for angle in range(0, -180, int(-360/n_plates)):
        plates_rotated[angle] = PlateCoords(plate_length, plate_thic,
                                            plate_width, gap,
                                            plts2_alpha, plts2_beta, angle)

        # Find coords of Electric field precise calculation zone
        upper_cube = np.array([np.min(plates_rotated[angle][0], axis=0),
                               np.max(plates_rotated[angle][0], axis=0)])

        lower_cube = np.array([np.min(plates_rotated[angle][1], axis=0),
                               np.max(plates_rotated[angle][1], axis=0)])

        # create mask for plates
        plates_mask[angle] = PlateFlags(upper_cube, lower_cube,
                                        plts2_alpha, plts2_beta, angle)

# %%
    eps = 1e-5


    for angle in range(0, -180, int(-360/n_plates)):
        # calculation loop
        step = 0
        t1 = time.time()
        while np.amax(np.abs(U1-U0)) > eps:
            step += 1
            U0 = np.copy(U)
            U = PDEstep(U, Uupper_plate, Ulower_plate, plates_mask, angle, edge_flag)
            if step > 1000: # wait until potential spreads to the center point
                U1 = np.copy(U)
    #            print(np.amax(np.abs(U1-U0)))

        print('Total number of steps = {}'.format(step))
        t2 = time.time()
        print("time needed for calculation: {:.5f} s\n".format(t2-t1))

        dirname='elecfield'
        try:
            os.mkdir(dirname) # create target Directory
            print("Directory " , dirname ,  " created ")
        except FileExistsError:
            pass

        Ex, Ey, Ez = np.gradient(-1*U, delta) # Ex, Ey, Ez
        fname='elecfield{}.dat'.format(angle)
        SaveElectricField(fname, Ex, Ey, Ez, plts2_alpha, plts2_beta, angle)

# %%
#    plot_contours(mesh_range_x, mesh_range_y, mesh_range_z, U, 30)
#    plot_quiver(mesh_range_x, mesh_range_y, mesh_range_z, Ex, Ey, Ez)
#    plot_quiver3d(x, y, z, Ex, Ey, Ez, 6)