# calculate magnetic fields arising from electrical current through wires of an
# arbitrary shape with the law of Biot-Savart

import numpy as np
import wire
from joblib import Parallel, delayed
#from itertools import repeat
import multiprocessing as mp
import time
from hibplib import importPFCoils
from hibpplotlib import *
import os

# %% define a fuction to calculate BiotSavart law--------------------------------
def BiotSavart(points, wires):
    '''
    calculate the magnetic field generated by currents flowing through wires
    :param wires: list of wire objects
    :param points: numpy array with x,y,z coordinates of n points
    :return: numpy array of n vectors representing the B field at given points
    '''
    if len(wires) == 0:
        print('no wires found!')
        return np.nan
    if len(points) == 0:
        print('no points found!')
        return np.nan

    print("found {} wire(s).".format(len(wires)))
    c = 0
    # generate list of IdL and r1 vectors from all wires
    for w in wires:
        c += 1
        _IdL, _r1 = w.IdL_r1
#        print("wire {} has {} segments".format(c, len(_IdL)))
        if c == 1:
            IdL = _IdL
            r1 = _r1
        else:
            IdL = np.vstack((IdL, _IdL))
            r1 = np.vstack((r1, _r1))
    print("total number of segments: {}".format(len(IdL)))
    print("number of field points: {}".format(len(points)))
    print("total number of calculations: {}".format(len(points)*len(IdL)))

    # now we have all segment vectors multiplied by the flowing current in IdL
    # and all vectors to the central points of the segments in r1

    def CalcBpoint(r,IdL,r1):
        r2 = r - r1
        r25 = np.linalg.norm(r2, axis=1)**3
        r3 = r2 / r25[:, np.newaxis]

        cr = np.cross(IdL, r3)

        # claculate sum of contributions from all current elements
        s = np.sum(cr, axis=0)
        return s

    # calculate vector B*1e7 for each point in space
    t1 = time.time()

    # calculate B at each point r
    # single processor
#    B = np.array([CalcBpoint(r, IdL, r1) * 1e-7 for r in points])

    # multiprocessing
    n_workers = mp.cpu_count()
    s = Parallel(n_jobs=n_workers)(delayed(CalcBpoint)(r, IdL, r1) for r in points)
    B = np.array(s)*1e-7

    t2 = time.time()
    print("time needed for calculation: {:.5f} s\n".format(t2-t1))

    return B


# %% Poloidal field calculation
def calcBpol(pf_coils, points, nx=3, ny=3):
    '''
    function calculates poloidal magnetic field from a toroidal coil
    pf_coils - dictionary with information about coils
    points - np array with points
    nx - number of filaments along x
    ny - number of filaments along y
    '''
    print('Calculating Poloidal Field')

    disc_len = 0.2  # discretisation length for wire [m]
    B = {}
    wires = []
    for coil in pf_coils.keys():

        print(coil)
        xc = pf_coils[coil][0]
        yc = pf_coils[coil][1]
        dx = pf_coils[coil][2]
        dy = pf_coils[coil][3]
        curr_tot = 1e6 # Total Current [A]
        curr = curr_tot/(nx*ny)
        # define toroidal angle
        nfi = 40 #number of steps along toroidal angle
        dfi = 2*np.pi/nfi
        fi = np.arange(0, 2*np.pi+dfi, dfi)

        # -> x cycle start ---------------------------------------------------
        for k_x in range(0, nx):
            # single wire is a circle in xz plane
            # set wire radius
            r = xc + dx/2 - k_x*dx/(nx-1)
            x = np.sin(fi) * r
            z = np.cos(fi) * r

            # -> y cycle start -----------------------------------------------
            for k_y in range(0, ny):
                # set y value
                y = np.full_like(x, yc + dy/2 - k_y*dy/(ny-1))
                # concatenate in one np array
                new_coil = np.c_[x, y, z]
                # create new wire object for calculation by Biotsavart script
                new_w = wire.Wire(path=new_coil, discretization_length=disc_len,
                                  current=curr)
                wires.append(new_w)
            # -> y cycle end -------------------------------------------------

        # -> x cycle end -----------------------------------------------------
        B_coil = BiotSavart(points, wires)
        B[coil] = B_coil
    return B, wires

# %% Toroidal field calculation
def calcBtor(points):
    '''
    function calculates toroidal field in points
    points - np array with x,y,z of n points
    '''
    print('Calculating Toroidal Field')
    n_coils = 16                # total number of coils in TOKAMAK
    coil_width = 0.196          # coil width [m]
    curr_tot = 0.484503*1e6    # Total Current in coil [A]

    n_xy = 4         # number of circuits in poloidal direction
    n_z = 4          # number of circuits in toroidal direction
    disc_len = 0.2   # discretisation length for wire [m]
    curr = curr_tot/(n_xy*n_z) # current in one circuit
    initial_coil_angle = (360/n_coils)*0.5  # initial toroidal angle of
                                            # the first coil

    # get coil inner and outer profile
    filename = 'coildata.dat'
    array = np.loadtxt(filename)  # [m]
    N_rows = array.shape[0]
    #array has only x and y columns, so soon we need to add a zero column for z
    # (because we have no z column in coildata)
    outer_coil_array = np.array(array[:, [2, 3]])
    inner_coil_array = np.array(array[:, [0, 1]])
    # Creating a missing "z" coordinate column
    z_column0 = [-coil_width*0.5] * N_rows
    wires = []
    # -> Coil cycle start ------------------------------------------------------
    for current_coil_number in range(n_coils):
        # toroidal angle of coil (0 means it is the first coil)
        z_column = z_column0
        coil_angle = initial_coil_angle + (360/n_coils) * (current_coil_number)

        # -> Toroidal cycle start ----------------------------------------------
        for k_z in range(1, n_z+1):
            new_coil =  np.c_[np.array(array[:, [2, 3]]), np.array(z_column)]

            # -> Poloidal cycle start ------------------------------------------
            for k_xy in range(1, n_xy+1):
                    new_coil[:, 0:2] = outer_coil_array - (k_xy/n_xy)* \
                                    (outer_coil_array - inner_coil_array)
                    # We add new wires for calculation by Biotsavart script
                    new_w = wire.Wire(path=new_coil,
                                      discretization_length=disc_len,
                                      current=curr).Rotate(axis=(0, 1, 0),
                                                            deg=coil_angle)
                    wires.append(new_w)

            # now we make a step in toroidal (z) direction:
            z_column = [-coil_width/2 + (coil_width/(n_z-1))*k_z] * N_rows
            # -> Poloidal cycle end---------------------------------------------

        # -> Toroidal cycle end ------------------------------------------------

    # -> Coil cycle end --------------------------------------------------------

    B = BiotSavart(points, wires)
    return B, wires

# %% Plasma Current Field
def ImportJpl(filename):
    ''' import plasma current distribution from Tokameq file'''
    with open(filename, 'r') as f:
        data = f.readlines()

    # R coordinate corresponds to X, Z coordinate corresponds to Y
    NrNz = []
    for i in data[2].strip().split():
        if i.isdigit():
            NrNz.append(i)
    Nx = int(NrNz[0]) + 1
    Ny = int(NrNz[1]) + 1

    for i in range(len(data)):
        if data[i].strip() == 'Current density J(r,z)':
            iJ = i
#            print(iJ)

    x_vals = [float(r) for r in data[iJ+1].strip().split()[1:]]
    x_vals = np.array(x_vals)

    J_data = [i.strip().split() for i in data[iJ+2:iJ+2+Ny]]
    J_vals = []
    y_vals = []
    for line in J_data:
        y_vals.append(float(line[0]))
        J_vals.append([float(j) for j in line[1:]])

    y_vals = np.array(y_vals)
    J_vals = np.array(J_vals)
    return J_vals, x_vals, y_vals

def calcBplasma(points, filename, CurrTot):
    ''' calculate plasma field in points
    filename - Tokameq file with Jpl distribution
    CurrTot - total plasma current in [MA] '''
    print('Calculating Plasma Field')
    J_vals, x_vals, y_vals = ImportJpl(filename)

    Jtot = np.sum(J_vals)  # total J, used for normalisation
    disc_len = 0.2  # discretisation length for wire [m]

    # define toroidal angle
    nfi = 40 #number of steps along toroidal angle

    dfi = 2*np.pi/nfi
    fi = np.arange(0, 2*np.pi+dfi, dfi)
    wires = []
    # -> x cycle start ---------------------------------------------------
    for i in range(x_vals.shape[0]):
        # single wire is a circle in xz plane
        # set wire radius as a value from x_vals
        x = np.sin(fi) * x_vals[i]
        z = np.cos(fi) * x_vals[i]
        # -> y cycle start -----------------------------------------------
        for j in range(y_vals.shape[0]):
            if J_vals[j,i] != 0.0:
                # set y value
                y = np.full_like(x, y_vals[j])
                # concatenate in one np array
                new_coil = np.c_[x, y, z]
                # create new wire object for calculation by Biotsavart script
                new_w = wire.Wire(path=new_coil, discretization_length=disc_len,
                                  current=1e6*CurrTot*J_vals[j,i]/Jtot)
                wires.append(new_w)
    B = BiotSavart(points, wires)
    return B, wires

#%%
def SaveMagneticField(fname, B, dirname="magfield"):
    ''' save magnetic field array to file
    '''
    try:
        os.mkdir(dirname) # create target Directory
        print("Directory " , dirname ,  " created ")
    except FileExistsError:
        pass

    open(dirname + '/' + fname, 'w').close()
    with open(dirname + '/' + fname, 'w') as myfile: # erases data from file before writing
        myfile.write('{} {} {} # volume corner 1\n'.format(volume_corner1[0],
                                                           volume_corner1[1],
                                                           volume_corner1[2]))
        myfile.write('{} {} {} # volume corner 2\n'.format(volume_corner2[0],
                                                           volume_corner2[1],
                                                           volume_corner2[2]))
        myfile.write('{} # resolution\n'.format(resolution))
        for i in range(B.shape[0]):
            myfile.write('{} {} {}\n'.format(B[i, 0], B[i, 1], B[i, 2]))
    print('Magnetic field saved, ' + fname)

#%%
if __name__ == '__main__':

    #if input("Recalculate magnetic fields [y/n]? ") == 'y':
    try:
        del B # delete previous magnetic field to avoid m
        print('\nDeleted previous magnetic field')
    except NameError:
        print('\nNo previous magnetic field found')

    Btor = 1.0 # Toroidal field [Tl]
    Ipl = 1.0  # Plasma current [MA]

    # Define grid points to caculate B
    resolution = 0.1    # [m]
    volume_corner1 = (0, -3.0, -0.2) # xmin ymin zmin [m]
    volume_corner2 = (3.5, 2.5, 0.2) # xmax ymax zmax [m]

    # create grid of points
    grid = np.mgrid[volume_corner1[0]:volume_corner2[0]:resolution,
                    volume_corner1[1]:volume_corner2[1]:resolution,
                    volume_corner1[2]:volume_corner2[2]:resolution]

    # create list of grid points
    points = np.vstack(map(np.ravel, grid)).T

    print('\n\nCalculating magnetic field with folowing params:\n' +
           ' Btor = {} [T]\n Ipl = {} [MA]\n'.format(Btor,Ipl)  +
           ' resolution = {}\n'.format(resolution) +
           ' volume_corner1 = {} [m]\n'.format(volume_corner1) +
           ' volume_corner2 = {} [m]\n'.format(volume_corner2))


    # calculate B field at given points
    B_tor, wires_tor = calcBtor(points)

    tokameq_file = '2MA_sn.txt' # Txt with plasma current calculated in Tokameq
    B_pl, wires_pl = calcBplasma(points, tokameq_file, Ipl)

    pf_coils = importPFCoils('PFCoils.dat')
    B_pol_dict, wires_pol = calcBpol(pf_coils, points)

    wires = wires_pol + wires_tor #+ wires_pl

    cutoff = 10.0
    Babs_tor = np.linalg.norm(B_tor, axis=1)
    B_tor[Babs_tor > cutoff] = [np.nan, np.nan, np.nan]

    Babs_pl = np.linalg.norm(B_pl, axis=1)
    B_pl[Babs_pl > cutoff] = [np.nan, np.nan, np.nan]

    fname='magfieldTor.dat'
#        SaveMagneticField(fname, B_tor)

    fname='magfieldPlasm{}.dat'.format(tokameq_file[13:16])
#        SaveMagneticField(fname, B_pl)

    B_check = B_tor*Btor + B_pl*Ipl # in B we will summatize filed values from all circuits

    for coil in B_pol_dict.keys():
        Babs_pol = np.linalg.norm(B_pol_dict[coil], axis=1)
        B_pol_dict[coil][Babs_pol > cutoff] = [np.nan, np.nan, np.nan]
        fname='magfield{}.dat'.format(coil)
#            SaveMagneticField(fname, B_pol_dict[coil])
        B_check += B_pol_dict[coil]

    print('\n\nCalculated magnetic field with folowing params:\n' +
          ' Btor = {} [T]\n Ipl = {} [MA]\n'.format(Btor,Ipl)  +
          ' resolution = {}\n'.format(resolution) +
          ' volume_corner1 = {} [m]\n'.format(volume_corner1) +
          ' volume_corner2 = {} [m]\n'.format(volume_corner2))

#    cutoff = 10.0
#    Babs = np.linalg.norm(B_check, axis=1)
#    B_check[Babs > cutoff] = [np.nan, np.nan, np.nan]

#%%
    plot_3d(B_check, wires, volume_corner1, volume_corner2, grid, resolution, cutoff=5.0)
#    plot_2d(B_check, points, plane='xy', cutoff=2, n_contours=50)
# %%
    # make an interpolation of B
#    x = np.arange(volume_corner1[0], volume_corner2[0], resolution)
#    y = np.arange(volume_corner1[1], volume_corner2[1], resolution)
#    z = np.arange(volume_corner1[2], volume_corner2[2], resolution)
#
#    t1_Binterp = time.time()
#    Bx = B[:, 0].reshape(grid.shape[1:])
#    By = B[:, 1].reshape(grid.shape[1:])
#    Bz = B[:, 2].reshape(grid.shape[1:])
#    Bx_interp = RegularGridInterpolator((x, y, z), Bx)
#    By_interp = RegularGridInterpolator((x, y, z), By)
#    Bz_interp = RegularGridInterpolator((x, y, z), Bz)
#    t2_Binterp = time.time()
