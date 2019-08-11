# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:57:47 2019

HIBP lib

@author: user
"""

import numpy as np
from collections import defaultdict
from matplotlib import path
from scipy.interpolate import RegularGridInterpolator
import os
import errno
import pickle as pc

# %% Runge-Kutt
def RungeKutt(q, m, RV, dt, E, B):
    '''
    Calculate one step using Runge-Kutta algorithm
    :param q: particle charge [Co]
    :param m: particle mass [kg]
    :param RV: 7 dimensial vector
           array[[x,y,z,vx,vy,vz,Flag]]
           Flag = 10 primary out of plasma
           Flag = 11 primary in plasma
           Flag = 20 secondary
    :param dt: time discretisation step
    :param E: values of electric field in current point
              np.array([Ex, Ey, Ez])
    :param B: values of magnetic field in current point
              np.array([Bx, By, Bz])
    :return: new RV

     V' = k(E + [VxB]) == K(E + np.cross(V,B)) == f
     r' = V == g

    V[n+1] = V[n] + (h/6)(m1 + 2m2 + 2m3 + m4)
    r[n+1] = r[n] + (h/6)(k1 + 2k2 + 2k3 + k4)
    m[1] = f(t[n], V[n], r[n])
    k[1] = g(t[n], V[n], r[n])
    m[2] = f(t[n] + (h/2), V[n] + (h/2)m[1], r[n] + (h/2)k[1])
    k[2] = g(t[n] + (h/2), V[n] + (h/2)m[1], r[n] + (h/2)k[1])
    m[3] = f(t[n] + (h/2), V[n] + (h/2)m[2], r[n] + (h/2)k[2])
    k[3] = g(t[n] + (h/2), V[n] + (h/2)m[2], r[n] + (h/2)k[2])
    m[4] = f(t[n] + h, V[n] + h*m[3], r[n] + h*k[3])
    k[4] = g(t[n] + h, V[n] + h*m[3], r[n] + h*k[3])

    E - np.array([Ex, Ey, Ez])
    B - np.array([Bx, By, Bz])
    '''
    k = q/m
    r = RV[0, :3]
    V = RV[0, 3:]

    # define equations of movement
    def f(E, V, B): return k*(E + np.cross(V, B))
    def g(V): return V

    ''' m1,k1 '''
    m1 = f(E, V, B)
    k1 = g(V)

    ''' m2,k2 '''
    fV2 = V + (dt / 2.) * m1
    gV2 = V + (dt / 2.) * m1

    m2 = f(E, fV2, B)
    k2 = g(gV2)

    ''' m3,k3 '''
    fV3 = V + (dt / 2.) * m2
    gV3 = V + (dt / 2.) * m2

    m3 =  f(E, fV3, B)
    k3 = g(gV3)

    ''' m4,k4 '''
    fV4 = V + dt * m3
    gV4 = V + dt * m3

    m4 = f(E, fV4, B)
    k4 = g(gV4)

    ''' all together! '''
    V = V + (dt / 6.) * (m1 + (2. * m2) + (2. * m3) + m4)
    r = r + (dt / 6.) * (k1 + (2. * k2) + (2. * k3) + k4)

    RV = np.hstack((r, V))

    return RV

# %%
def importPFCoils(filename):
    ''' import a dictionary with poloidal field coils parameters
    {'NAME': (x center, y center, width along x, width along y [m],
               current [MA-turn], N turns)}
    Andreev, VANT 2014, No.3
    '''
    d = defaultdict(list)
    with open(filename) as f:
        for line in f:
            if line[0] == '#':
                continue
            lineList = line.split(', ')
            key, val = lineList[0], tuple([float(i) for i in lineList[1:]])
            d[key] = val
    return d

def ImportPFcur (filename, pf_coils):
    '''
    Creates dictionary. containing coils names and their currents
    :param filename: Tokameqs filename
    :param coils: coil dict (we only take keys)
    :return: PF dictianary with currents
    '''
    with open(filename, 'r') as f:
        data = f.readlines()  # read tokameq file
    PF_dict = {}              # Here we will store coils names and currents
    pf_names = list(pf_coils) # get coils names
    l = 0 # will be used for getting correct coil name
    for i in range(len(data)):
        if data[i].strip() == 'External currents:':
            k = i + 2  # skip 2 lines and read from the third
            break
    while float(data[k].strip().split()[3]) != 0:
        key = pf_names[l]
        val = data[k].strip().split()[3]
        PF_dict[key] = val
        k += 1
        l += 1

    return PF_dict

# %%
def Translate(input_array, xyz):
    '''
    move the vector in space
    :param xyz: 3 component vector that describes translation in x,y,z direction
    :return: translated input_array
    '''
    if input_array is not None:
        input_array += np.array(xyz)

    return input_array

# %%
def Rotate(input_array, axis=(1, 0, 0), deg=0):
    '''
    rotate vector around given axis by deg degrees
    :param axis: axis of rotation
    :param deg: angle in degrees
    :return: rotated input_array
    '''
    if input_array is not None:
        n = axis
        ca = np.cos(np.radians(deg))
        sa = np.sin(np.radians(deg))
        R = np.array([[n[0]**2*(1-ca)+ca, n[0]*n[1]*(1-ca)-n[2]*sa, n[0]*n[2]*(1-ca)+n[1]*sa],
                      [n[1]*n[0]*(1-ca)+n[2]*sa, n[1]**2*(1-ca)+ca, n[1]*n[2]*(1-ca)-n[0]*sa],
                      [n[2]*n[0]*(1-ca)-n[1]*sa, n[2]*n[1]*(1-ca)+n[0]*sa, n[2]**2*(1-ca)+ca]])
        input_array = np.dot(input_array, R.T)

    return input_array

# %% Intersection check functions

def LinePlaneIntersect(planeNormal, planePoint, rayDirection, rayPoint, eps=1e-6):
    ''' function returns intersection point between plane and vector
    '''
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < eps:
        raise RuntimeError("no intersection or line is within plane")

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi

def LineSegmentsIntersect(A, B, C, D): # doesn't work with collinear case
    ''' function returns true if line segments AB and CD intersect
    '''
    def order(A, B, C):
        ''' If counterclockwise return True
            If clockwise return False '''
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    # Return true if line segments AB and CD intersect
    return order(A, C, D) != order(B, C, D) and order(A, B, C) != order(A, B, D)

def SegmentPolygonIntersection(polygon_coords, segment_coords):
    ''' check segment and polygon intersection'''
    polygon_normal = np.cross(polygon_coords[2, 0:3]-polygon_coords[0, 0:3],\
                              polygon_coords[1, 0:3]-polygon_coords[0, 0:3])
    polygon_normal = polygon_normal/np.linalg.norm(polygon_normal)

    intersect_coords = LinePlaneIntersect(polygon_normal, polygon_coords[2, 0:3], \
                          segment_coords[1, 0:3]-segment_coords[0, 0:3], \
                          segment_coords[0, 0:3])
    i = np.argmax(polygon_normal)
    polygon_coords = np.delete(polygon_coords, i, 1)
    intersect_coords = np.delete(intersect_coords, i, 0)
    p = path.Path(polygon_coords)
    return p.contains_point(intersect_coords)


# %%
def PlacePlate(r_dict, plts_angles, dirname = 'elecfield'):
    '''
    read plate's shape and angle parametres from provided file (should lbe in the same directory)
    return: plate's geometry array
    :param fname: filename
    :param xyz: coordinate array of plates centre
    '''

    Plates_Geom = {}

    dirname = dirname + '/' + 'alpha2_{}_beta2_{}'.format(int(plts_angles[0]),
                                int(plts_angles[1]))

    for filename in os.listdir(dirname):
        if filename.endswith(".dat"):
            try:
                print('Placing {} plate...'.format(filename[-6:-4]))
                xyz = r_dict[filename[-6:-4]]
            except KeyError:
                print('Plate unrecognised. Aborting plate placement...')
                raise Exception('PlateUnrecognised!')

            with open(dirname + '/' + filename, 'r') as f:
                geom_list = [float(i) for i in f.readline().replace(' # plate\'s length, thic, width and gap\n','').split(' ')]
                angle_line = [float(i) for i in f.readline().replace(' # plate\'s alpha, beta and gamma angle\n','').split(' ')]

            alpha, beta, gamma = angle_line

            # Geometry rotated in system based on central point between plates
            plate_length = geom_list[0]
            plate_width = geom_list[2]
            gap = geom_list[3]
            # Upper plate
            UP5 =  np.array([-plate_length/2., gap/2., plate_width/2.])
            UP6 =  np.array([-plate_length/2., gap/2., -plate_width/2.])
            UP7 =  np.array([plate_length/2., gap/2., -plate_width/2.])
            UP8 =  np.array([plate_length/2., gap/2., plate_width/2.])
            UP_full = np.array([UP5, UP6, UP7, UP8]) # coordinates of upper's plate edges
            UP_rotated_translated = UP_full.copy()
            for UP in range(UP_rotated_translated.shape[0]):
                UP_rotated_translated[UP, :] = Rotate(UP_rotated_translated[UP, :], axis=(1, 0, 0), deg=gamma)
                UP_rotated_translated[UP, :] = Rotate(UP_rotated_translated[UP, :], axis=(0, 0, 1), deg=alpha)
                UP_rotated_translated[UP, :] = Rotate(UP_rotated_translated[UP, :], axis=(0, 1, 0), deg=beta)
                UP_rotated_translated[UP, :] = Translate(UP_rotated_translated[UP, :], (xyz[0], xyz[1], xyz[2]))
            UP_plane_normal = np.cross(UP5-UP6, UP6-UP7)

            # lower plate
            LP5 =  np.array([-plate_length/2., -gap/2., plate_width/2.])
            LP6 =  np.array([-plate_length/2., -gap/2., -plate_width/2.])
            LP7 =  np.array([plate_length/2., -gap/2., -plate_width/2.])
            LP8 =  np.array([plate_length/2., -gap/2., plate_width/2.])
            LP_full = np.array([LP5, LP6, LP7, LP8]) # coordinates of lower's plate edges
            LP_rotated_translated = LP_full.copy()
            for LP in range(LP_full.shape[0]):
                LP_rotated_translated[LP, :] = Rotate(LP_rotated_translated[LP, :], axis=(1, 0, 0), deg=gamma)
                LP_rotated_translated[LP, :] = Rotate(LP_rotated_translated[LP, :], axis=(0, 0, 1), deg=alpha)
                LP_rotated_translated[LP, :] = Rotate(LP_rotated_translated[LP, :], axis=(0, 1, 0), deg=beta)
                LP_rotated_translated[LP, :] = Translate(LP_rotated_translated[LP, :], (xyz[0], xyz[1], xyz[2]))
            LP_plane_normal = np.cross(LP5-LP6, LP6-LP7)

#            plates_normals = np.array([UP_plane_normal, LP_plane_normal])
#            edges_coords = np.array([UP_rotated_translated, LP_rotated_translated])
            Plates_Geom[filename[-6:-4]] = np.array([UP_plane_normal, LP_plane_normal,
                                                     UP_rotated_translated,
                                                     LP_rotated_translated])

    return Plates_Geom

# %%
def ReadElecField(r_dict, plts_angles, dirname = 'elecfield'):
    '''
    read plate's shape and angle parametres along with electric field values
    from provided file (should lbe in the same directory)
    return: intrepolator function for electric field
    :param fname: filename
    :param r_dict: dict (key: name of plate,
                         value: coordinate array of plates centre)
    '''
    E=[]

    dirname = dirname + '/' + 'alpha2_{}_beta2_{}'.format(int(plts_angles[0]),
                                int(plts_angles[1]))

    for filename in os.listdir(dirname):
        if filename.endswith(".dat"):
            try:
                print('Reading {} electric field...'.format(filename[-6:-4]))
                xyz = r_dict[filename[-6:-4]]
            except KeyError:
                print('Plate unrecognised. Aborting...')
                raise Exception('PlateUnrecognised!')

            with open(dirname + '/' + filename, 'r') as f:
                geom_list = [float(i) for i in f.readline().replace(' # plate\'s length, thic, width and gap\n','').split(' ')]
                angle_line = [float(i) for i in f.readline().replace(' # plate\'s alpha, beta and gamma angle\n','').split(' ')]
                first_line = [int(i) for i in f.readline().replace(' # number of dots (x,y,z)\n','').split(' ')]
                second_line = [float(i) for i in f.readline().replace(' # border x, border y, border z, delta\n','').replace('\n','').split(' ')]
                if round(angle_line[0],1) == round(plts_angles[0]*(180/np.pi),1)  and \
                   round(angle_line[1],1) == round(plts_angles[1]*(180/np.pi),1):
                    print('\nERROR: Angles do not match!\nRecalculate electric field with desired angles or change plates\' angles in test_class.py to match ones used for initial  electric field caluclation\n')
                    raise
                Ex = np.zeros((first_line[0], first_line[1], first_line[2]))
                Ey = np.zeros((first_line[0], first_line[1], first_line[2]))
                Ez = np.zeros((first_line[0], first_line[1], first_line[2]))

                for i in range(first_line[0]):
                    for j in range(first_line[1]):
                        for k in range(first_line[2]):
                            line = [float(l) for l in f.readline().replace('\n','').split(' ')]
                            Ex[i,j,k] = line[0]
                            Ey[i,j,k] = line[1]
                            Ez[i,j,k] = line[2]


            mesh_range_x = np.arange(-second_line[0]/2., second_line[0]/2., second_line[3])
            mesh_range_y = np.arange(-second_line[1]/2., second_line[1]/2., second_line[3])
            mesh_range_z = np.arange(-second_line[2]/2., second_line[2]/2., second_line[3])


            # make interpolation for Ex, Ey, Ez
            Ex_interp = RegularGridInterpolator((mesh_range_x + xyz[0], mesh_range_y + xyz[1], mesh_range_z + xyz[2]), Ex)
            Ey_interp = RegularGridInterpolator((mesh_range_x + xyz[0], mesh_range_y + xyz[1], mesh_range_z + xyz[2]), Ey)
            Ez_interp = RegularGridInterpolator((mesh_range_x + xyz[0], mesh_range_y + xyz[1], mesh_range_z + xyz[2]), Ez)
            E_read = [Ex_interp, Ey_interp, Ez_interp]
        E.append(E_read)


    return E

# %%
def ReadMagField(Btor,Ipl, PF_dict, dirname ='magfield'):
    '''
    read magnetic field values from provided file (should lbe in the same directory)
    return: list of intrepolator functions for Bx, By, Bz
    :param dirname: name of directory with magfield dats
    '''

    B_full={}
    for filename in os.listdir(dirname):
        if filename.endswith(".dat"):
            with open(dirname + '/' + filename, 'r') as f:
                volume_corner1 = [float(i) for i in f.readline().replace(' # volume corner 1\n','').split(' ')]
                volume_corner2 = [float(i) for i in f.readline().replace(' # volume corner 2\n','').split(' ')]
                resolution = float(f.readline().replace(' # resolution\n',''))
                if 'Tor' in filename and Btor != 0:
                    print('Reading toroidal magnetic field...')
                    B_read = np.loadtxt(dirname + '/' + filename,skiprows=3) * Btor

                elif 'Plasm' in filename:
                    print('Reading plasma field...')
                    B_read = np.loadtxt(dirname + '/' + filename,skiprows=3) * Ipl

                else:
                    print('Reading {} magnetic field...'.format(filename[-7:-4]))
                    Icir = float(PF_dict[filename[-7:-4]])
                    B_read = np.loadtxt(dirname + '/' + filename, skiprows=3) * Icir

                B_full[filename[-7:-4]] = B_read

    # create grid of points
    grid = np.mgrid[volume_corner1[0]:volume_corner2[0]:resolution,
                    volume_corner1[1]:volume_corner2[1]:resolution,
                    volume_corner1[2]:volume_corner2[2]:resolution]

    B = np.zeros(B_read.shape)
    for key in B_full.keys():
        B += B_full[key]
#
#    cutoff = 10.0
#    Babs = np.linalg.norm(B, axis=1)
#    B[Babs > cutoff] = [np.nan, np.nan, np.nan]
    # make an interpolation of B
    x = np.arange(volume_corner1[0], volume_corner2[0], resolution)
    y = np.arange(volume_corner1[1], volume_corner2[1], resolution)
    z = np.arange(volume_corner1[2], volume_corner2[2], resolution)
    Bx = B[:, 0].reshape(grid.shape[1:])
    By = B[:, 1].reshape(grid.shape[1:])
    Bz = B[:, 2].reshape(grid.shape[1:])
    Bx_interp = RegularGridInterpolator((x, y, z), Bx)
    By_interp = RegularGridInterpolator((x, y, z), By)
    Bz_interp = RegularGridInterpolator((x, y, z), Bz)

    B = [Bx_interp, By_interp, Bz_interp]

    return B

# %%
def ReturnElecField(xyz, Ein, U):
    '''
    take dot and try to interpolate electiric fields in it
    return: interpolated Electric field
    :param Ein: list of interpolants for Ex, Ey, Ez
    '''
    Eout = np.zeros(3)
    for i in range(len(Ein)):
        try:
            Eout[0] += Ein[i][0](xyz)*U[i]
            Eout[1] += Ein[i][1](xyz)*U[i]
            Eout[2] += Ein[i][2](xyz)*U[i]
        except:
            continue

    return Eout

# %%
def SaveTrajList(traj_list, Btor, Ipl, r_aim, dirname='output'):
    ''' function saves list of Traj objects to pickle file
    :param traj_list: list of trajectories
    '''

    if len(traj_list) == 0:
        print('traj_list empty!')
        return

    Ebeam_list = []
    UA2_list = []

    for traj in traj_list:
        Ebeam_list.append(traj.Ebeam)
        UA2_list.append(traj.UA2)

    dirname = dirname + '/' + 'B{}_I{}'.format(int(Btor), int(Ipl))

    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname, 0o700)
            print("Directory " , dirname ,  " created ")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    fname = dirname + '/' + \
                'E{}-{}_UA2{}-{}_alpha{}_beta{}_x{}y{}z{}.pkl'.format(int(min(Ebeam_list)),
                  int(max(Ebeam_list)), int(min(UA2_list)), int(max(UA2_list)),
                  int(round(traj.alpha)),
                  int(round(traj.beta)),
                  int(r_aim[0,0]*100), int(r_aim[0,1]*100), int(r_aim[0,2]*100))

    with open(fname, 'wb') as f:
        pc.dump(traj_list, f, -1)

    print('\nSAVED LIST: \n' + fname)
#%%


def save_png(fig, name, save_dir = 'Results/Grids'):
    """
    Saves picture as name.png
    
    Args:
    :fig - array of figures to save
    :name - array of picture names
    :save_dir - directory used to store results
    """

    # check wether directory exist and if not - create one
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        print('LOG: {} directory created'.format(save_dir))
    print('LOG: Saving pictures to {}'.format(save_dir+'/'))
    for fig, name in zip(fig, name):
        # save fig with tight layout
        fig_savename = str(name + '.png')
        fig.savefig(save_dir + '/'+ fig_savename, bbox_inches='tight')
        print('LOG: Figure ' + fig_savename + ' saved')
#%%


def ReadTrajList(fname, dirname='output'):
    '''
    import list of Traj objects from .pkl file
    '''
    with open(dirname + '/'+fname, 'rb') as f:
        traj_list = pc.load(f)
    return traj_list