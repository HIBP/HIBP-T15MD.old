import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
from matplotlib.patches import Rectangle
from hibplib import *
import pickle as pc
import os

# %% define class for trajectories
class Traj():
    def __init__(self, q, m, Ebeam, r0, alpha, beta, UA2=0.0, UB2=0.0, dt=1e-7):
        ''' class for trajectories
        q - particle charge [Co]
        m - particle mass [kg]
        Ebeam - beam energy [keV]
        r0 - initial position [m]
        alpha - injection angle in XY plane [rad]
        beta - injection angle in XZ plane [rad]
        UA2 - voltage on A2 deflecting plates [kV]
        dt - timestep for RK algorithm [s]
        '''
        self.q = q
        self.m = m
        self.Ebeam = Ebeam
        Vabs = np.sqrt(2 * Ebeam * 1.602176487E-16 / m)
        V0 = np.array([-Vabs * np.cos(alpha) * np.cos(beta),
                       -Vabs * np.sin(alpha),
                       Vabs * np.cos(alpha) * np.sin(beta)])
        self.alpha = alpha
        self.beta = beta
        self.UA2 = UA2
        self.UB2 = UB2
        self.RV0 = np.array([np.hstack((r0, V0))])  # initial condition

        self.RV_Prim = self.RV0  # array to contain r,V for the whole trajectory
        self.tag_Prim = [1]

        self.RV_Sec = np.array([[]])
        self.tag_Sec = [2]

        # list to contain RV of the whole fan
        self.Fan = []
        # time step for primary orbit
        self.dt1 = dt
        # time step for secondary orbit
        self.dt2 = dt
        self.IsAimXY = False
        self.IsAimZ = False
        self.IntersectGeometry = False
        self.IntersectGeometrySec = False

    def PassPrim(self, RV0, E, B_interp):
        ''' passing primary trajectory from initial point self.RV0
        '''
#        print('Passing primary trajectory')
        t = 0.
        dt = self.dt1
        tmax = 1.  # [sec]
        elon = 1.7  # plasma size in along Y [m]
        r_plasma = 0.9  # plasma size along X [m]
        RV_old = self.RV0  # initial position
        RV = self.RV0  # array to collect all r,V
        tag_column = [10]

        while t <= tmax:
            r = RV_old[0, :3]

            # Electric field
            E_currently = ReturnElecField(r, E, [self.UA2, self.UB2])

            # Magnetic field
            try:
                Bx_interp, By_interp, Bz_interp = B_interp[0], B_interp[1], B_interp[2]
                B_currently = np.c_[Bx_interp(r), By_interp(r), Bz_interp(r)]
            except ValueError:
#                print('Btor Out of bounds for primaries')
                break

            RV_new = RungeKutt(self.q, self.m, RV_old, dt, E_currently, B_currently)

            if (LineSegmentsIntersect(RV_new[0, 0:2], RV_old[0, 0:2],
                                      chamb_ent[0], chamb_ent[1])) or \
               (LineSegmentsIntersect(RV_new[0, 0:2], RV_old[0, 0:2],
                                      chamb_ent[2], chamb_ent[3])):
#                print('Intersection with chamber appeared')
                self.IntersectGeometry = True
                break

#            if (SegmentPolygonIntersection(UP_rotated_translated, np.array([RV_new[0, 0:3],RV_old[0, 0:3]]))) or \
#               (SegmentPolygonIntersection(LP_rotated_translated, np.array([RV_old[0, 0:3],RV_new[0, 0:3]]))):
            if (LineSegmentsIntersect(RV_new[0, 0:2], RV_old[0, 0:2],
                                      A2_edges[0][0, 0:2], A2_edges[0][3, 0:2])) or \
               (LineSegmentsIntersect(RV_new[0, 0:2], RV_old[0, 0:2],
                                      A2_edges[1][0, 0:2], A2_edges[1][3, 0:2])):
#                print('Intersection with plates appeared')
                self.IntersectGeometry = True
                break

            RV = np.vstack((RV, RV_new))

            # eliptical radius of particle
            # 1.5 m is the major radius of a torus
            r_xy = np.sqrt((RV_new[0, 0] - 1.5)**2 + (RV_new[0, 1] / elon)**2)
            if r_xy <= r_plasma:
                tag_column = np.hstack((tag_column, 11))
            else:
                tag_column = np.hstack((tag_column, 10))

            RV_old = RV_new
            t = t + dt

    #        print('t = ', t)

        self.RV_Prim = RV
        self.tag_Prim = tag_column

    def PassSec(self, RV0, r_aim, E, B_interp, eps_xy=5e-3, eps_z=5e-3): # eps_xy=5e-3, eps_z=5e-3
        ''' passing secondary trajectory from initial point RV0 to point r_aim
            with accuracy eps
        '''
#        print('Passing secondary trajectory')
        t = 0
        dt = self.dt2
        tmax = dt*100.
        RV_old = RV0  # initial position
        RV = RV0  # array to collect all r,V
        tag_column = [20]

        while t <= tmax:  # to witness curls initiate tmax as 1 sec
            r = RV_old[0, :3]

            # Electric field
            E_currently = ReturnElecField(r, E, [self.UA2, self.UB2])

            # Magnetic field
            try:
                Bx_interp, By_interp, Bz_interp = B_interp[0], B_interp[1], B_interp[2]
                B_currently = np.c_[Bx_interp(r), By_interp(r), Bz_interp(r)]
            except ValueError:
#                print('Btor Out of bounds for secondaries')
                break

            RV_new = RungeKutt(2*self.q, self.m, RV_old, dt, E_currently, B_currently)

            if (LineSegmentsIntersect(RV_new[0, 0:2], RV_old[0, 0:2],
                                      chamb_ext[0], chamb_ext[1])) or \
               (LineSegmentsIntersect(RV_new[0, 0:2], RV_old[0, 0:2],
                                      chamb_ext[2], chamb_ext[3])):
#                print('Secondary intersected chamber exit')
                self.IntersectGeometrySec = True


            if (RV_new[0, 0] > r_aim[0, 0]) & (RV_new[0, 1] < 1.0):
                # YZ plane
                planeNormal = np.array([1, 0, 0])
                planePoint = r_aim[0]
                rayDirection = RV_new[0, :3] - RV_old[0, :3]
                rayPoint = RV_new[0, :3]
                r_intersect = LinePlaneIntersect(planeNormal, planePoint,
                                                 rayDirection, rayPoint)
                RV_new[0, :3] = r_intersect
                RV = np.vstack((RV, RV_new))
                # check XY plane
                if (np.linalg.norm(RV_new[0, :2] - r_aim[0, :2]) <= eps_xy):
#                    print('aim XY!')
                    self.IsAimXY = True
                # check XZ plane
                if (np.linalg.norm(RV_new[0, [0, 2]] - r_aim[0, [0, 2]]) <= eps_z):
#                    print('aim Z!')
                    self.IsAimZ = True
                break
            else:
                RV_old = RV_new
                t = t + dt
                RV = np.vstack((RV, RV_new))

            tag_column = np.hstack((tag_column, 20))
    #        print('t = ', t)

        self.RV_Sec = RV
        self.tag_Sec = tag_column

    def PassFan(self, r_aim, E, B_interp):
        ''' passing fan from initial point self.RV0
        '''
#        print('Passing fan of trajectories')
        self.PassPrim(self.RV0, E, B_interp)

        # list of initial points of secondary trajectories
        RV0_Sec = self.RV_Prim[(self.tag_Prim == 11)]

        list_sec = []
        for RV02 in RV0_Sec:
            RV02 = np.array([RV02])
            self.PassSec(RV02, r_aim, E, B_interp)
            list_sec.append(self.RV_Sec)

        self.Fan = list_sec

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
    try:
        os.mkdir(dirname) # create target Directory
        print("Directory " , dirname ,  " created ")
    except FileExistsError:
        pass

    fname = dirname + '/' + \
                'E{}-{}_UA2{}-{}_alpha{}_beta{}_x{}y{}z{}.pkl'.format(int(min(Ebeam_list)),
                  int(max(Ebeam_list)), int(min(UA2_list)), int(max(UA2_list)),
                  int(round(traj.alpha*180/np.pi)),
                  int(round(traj.beta*180/np.pi)),
                  int(r_aim[0,0]*100), int(r_aim[0,1]*100), int(r_aim[0,2]*100))

    with open(fname, 'wb') as f:
        pc.dump(traj_list, f, -1)

    print(fname + ' ***SAVED')

def ReadTrajList(fname, dirname='output'):
    '''
    import list of Traj objects from .pkl file
    '''
    with open(fname, 'rb') as f:
        traj_list = pc.load(f)
    return traj_list

# %%
''' MAIN '''

if __name__ == '__main__':
    # toroidal field on axis
    Btor = 1.0  # [T]
    Ipl = 1.0  # Plasma current [MA]
    q = 1.60217662e-19  # electron charge [Co]
    m_Tl = 204.3833 * 1.6605e-27  # Tl ion mass [kg]

    # initial beam energy range
    dEbeam = 20.
    Ebeam_range = np.arange(80.,400. + dEbeam, dEbeam)  # [keV]

    #A2 plates voltage
    dUA2 = 10.0
    UA2_range = np.arange(-30., 60. + dUA2, dUA2)  # [kV]

    #B2 plates voltage
    UB2 = 10.0  # [kV]
    dUB2 = 15.0  # [kV/m]

    # alpha and beta angles of primary beamline
    alpha_prim = 25.*(np.pi/180)  # rad
    beta_prim = -10.*(np.pi/180)  # rad
    gamma_prim = -90.*(np.pi/180)  # rad
    angles_prime = np.array([alpha_prim, beta_prim, gamma_prim])

    # coordinates of injection pipe [m]
    xpatr = 1.5 + 0.726
    ypatr = 1.064
    zpatr = 0.0
#    xpatr = 2.45
#    ypatr = 0.4
#    zpatr = 0.0
    # distance from injection pipe to Alpha2 plates
    dist_A2 = 0.3  # [m]
    dist_B2 = 0.45  # [m]
    # distance from injection pipe to initial piont of trajectory [m]
    dist_0 = 0.6

    # coordinates of Alpha2 plates
    xA2 = xpatr + dist_A2*np.cos(alpha_prim)*np.cos(beta_prim)
    yA2 = ypatr + dist_A2*np.sin(alpha_prim)
    zA2 = zpatr - dist_A2*np.cos(alpha_prim)*np.sin(beta_prim)
    rA2 = np.array([xA2, yA2, zA2])

    # coordinates of Alpha2 plates
    xB2 = xpatr + dist_B2*np.cos(alpha_prim)*np.cos(beta_prim)
    yB2 = ypatr + dist_B2*np.sin(alpha_prim)
    zB2 = zpatr - dist_B2*np.cos(alpha_prim)*np.sin(beta_prim)
    rB2 = np.array([xB2, yB2, zB2])

    # coordinates of initial point of trajectory [m]
    x0 = xpatr + dist_0*np.cos(alpha_prim)*np.cos(beta_prim)
    y0 = ypatr + dist_0*np.sin(alpha_prim)
    z0 = zpatr - dist_0*np.cos(alpha_prim)*np.sin(beta_prim)
    r0 = np.array([x0, y0, z0])

    # timestep [sec]
    dt = 1e-7

    r_aim = np.array([[2.6, -0.2, 0.]])
#    r_aim = np.array([[2.75, -0.5, 0.]])

    # chamber entrance coordinates
#    chamb_ent = [(2.01,1.08),(1.86,1.31),(2.385,0.52),(2.19,0.82)]
    chamb_ent = [(2.01, 1.102), (1.99, 1.265), (2.211, 0.937), (2.339, 0.746)]
    chamb_ext = [(2.34, -0.46), (2.34, -1.), (2.34, 0.46), (2.34, 0.5)]

#%%
    '''
    Electric field part
    '''
    fname = 'elecfieldA2.dat'
    A2_normals, A2_edges = PlacePlate(fname, rA2)
    EA2 = ReadElecField(fname, rA2, angles_prime)

    fname = 'elecfieldB2.dat'
    B2_normals, B2_edges = PlacePlate(fname, rB2)
    EB2 = ReadElecField(fname, rB2, angles_prime)
    E = [EA2, EB2]

    '''
    Magnetic field part
    '''
    pf_coils = importPFCoils('PFCoils.dat')

    PF_dict = ImportPFcur('{}MA_sn.txt'.format(int(abs(Ipl))), pf_coils)
    if not 'B' in locals():
        dirname='magfield'
        B = ReadMagField(Btor, Ipl, PF_dict, dirname)

# %%
    ''' Scan cycle '''
    # list of trajectores that hit r_aim
    traj_list = []

    for Ebeam in Ebeam_range:
        for UA2 in UA2_range:
            print('\n\nE = {} keV; UA2 = {} kV\n'.format(Ebeam, UA2))

            try:
                # create new trajectory object
                traj1 = Traj(q, m_Tl, Ebeam, r0, alpha_prim, beta_prim, UA2, UB2, dt)

                while not traj1.IsAimZ:
                    traj1.UB2, traj1.dt1, traj1.dt2 = UB2, dt, dt

                    # pass fan of trajectories
                    traj1.PassFan(r_aim, E, B)
                    # reset flags in order to let the algorithm work properly
                    traj1.IsAimXY, traj1.IsAimZ = False, False
                    traj1.IntersectGeometrySec = False

#                    if traj1.IntersectGeometry:
#                        break

                    # find which secondaries are higher/lower than r_aim
                    signs = np.array([np.sign(np.cross(RV[-1, :3], r_aim[0, :])[-1])
                                      for RV in traj1.Fan])  # -1 higher, 1 lower
                    are_higher = np.argwhere(signs == -1)
                    are_lower = np.argwhere(signs == 1)
                    if are_higher.shape[0] == 0:
                        print('Aim is too HIGH along Y!')
#                        traj_list.append(traj1)
                        break
                    elif are_lower.shape[0] == 0:
                        print('Aim is too LOW along Y!')
#                        traj_list.append(traj1)
                        break
                    else:
                        n = int(are_higher[-1])  # find one which is higher
                    RV_old = np.array([traj1.Fan[n][0]])

                    # find secondary, which goes directly into r_aim
                    traj1.dt1 = traj1.dt1/2.

                    while True:
                        try:
                            Bx_interp, By_interp, Bz_interp = B[0], B[1], B[2]
                            # make a small step alomg primary trajectory
                            r = RV_old[0, :3]
                            RV_new = RungeKutt(q, m_Tl, RV_old, traj1.dt1, np.c_[0, 0, 0],
                                               np.c_[Bx_interp(r), By_interp(r), Bz_interp(r)])
                        except:
                            print('MyError: concatination')
                            break
                        # pass new secondary trajectory
                        traj1.PassSec(RV_new, r_aim, E, B)

                        if traj1.IsAimXY:
                            break

                        # check if the new secondary trajectory is lower than r_aim
                        if np.sign(np.cross(traj1.RV_Sec[-1, :3], r_aim[0, :])[-1]) > 0:
                            # if lower, halve the timestep and try one more time
                            traj1.dt1 = traj1.dt1/2.
                            print('dt1={}'.format(traj1.dt1))
                            if traj1.dt1 < 1e-10:
                                print('dt too small')
                                break
                        else:
                            # if higher, continue steps along the primary
                            RV_old = RV_new

                    if not traj1.IsAimZ:
#                        print('UB2 OLD = {:.2f}, z_aim - z_curr = {:.4f}'.format(UB2, r_aim[0,2]-traj1.RV_Sec[-1, 2]))
                        UB2 = UB2 - dUB2*(r_aim[0, 2] - traj1.RV_Sec[-1, 2])
#                        print('UB2 NEW = {:.2f}'.format(UB2))
                    else:
                        break

            except (IndexError, ValueError, NameError) as err:
                print('ERROR : ', err)
                pass

            if traj1.IsAimXY and traj1.IsAimZ:
                traj_list.append(traj1)
                print('trajectory saved, UB2={:.2f} kV'.format(traj1.UB2))

# %%
    plt.close('all')
    traj_list_passed = []  # list of trajs that passed geometry limitations
    if len(traj_list) != 0:
        for traj in traj_list:
            if not traj.IntersectGeometrySec:
#                plot_fan([traj], r_aim, A2_edges, B2_edges, traj.Ebeam, traj.UA2, Btor, Ipl)
                traj_list_passed.append(traj)
        print('found {} trajectories'.format(len(traj_list)))
    else:
        print('There is nothing to plot')

# %%
    plot_grid(traj_list_passed, r_aim, Btor, Ipl, marker_E='')
    plot_scan(traj_list_passed, r_aim, 220., Btor, Ipl)

# %%
#    SaveTrajList(traj_list_passed, Btor, Ipl, r_aim)
