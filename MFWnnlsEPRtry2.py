import numpy as np
from numpy.linalg import eig
import cmath
import math
import sys
from functools import reduce
#import pandas as pd
#import pylab as pl
import array
#from scipy.fftpack import fft, ifft
from scipy.optimize import nnls


def random_unit_vector():
    phi = 2.0 * math.pi * np.random.random()
    z = 2.0 * np.random.random() - 1.0
    r = math.sqrt(1.0 - z * z)
    return np.array([r * math.cos(phi), r * math.sin(phi), z])


class Rotation:
    """ 
    * Rotation : provides a representation for 3D space rotations
    * using euler angles (ZX'Z'' convention) or rotation matrices
    """

    def _euler2mat_z1x2z3(self, z1=0, x2=0, z3=0):
        cosz1 = math.cos(z1)
        sinz1 = math.sin(z1)
        Z1 = np.array(
            [[cosz1, -sinz1, 0],
             [sinz1, cosz1, 0],
             [0, 0, 1]])

        cosx = math.cos(x2)
        sinx = math.sin(x2)
        X2 = np.array(
            [[1, 0, 0],
             [0, cosx, -sinx],
             [0, sinx, cosx]])

        cosz3 = math.cos(z3)
        sinz3 = math.sin(z3)
        Z3 = np.array(
            [[cosz3, -sinz3, 0],
             [sinz3, cosz3, 0],
             [0, 0, 1]])

        return reduce(np.dot, [Z1, X2, Z3])

    def _mat2euler(self, M):
        M = np.asarray(M)
        try:
            sy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            sy_thresh = _FLOAT_EPS_4
        r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
        sy = math.sqrt(r31 * r31 + r32 * r32)
        if sy > sy_thresh:
            x2 = math.acos(r33)
            z1 = math.atan2(r13, -r23)
            z3 = math.atan2(r31, r32)
        else:
            x2 = 0
            z3 = 0
            z1 = math.atan2(r21, r22)
        return (z1, x2, z3)

    def _init_from_angles(self, z1, x2, z3):
        self._z1, self._x2, self._z3 = z1, x2, z3
        self._M = self._euler2mat_z1x2z3(self._z1, self._x2, self._z3)

    def _init_from_matrix(self, matrix):
        self._M = np.asarray(matrix)
        self._z1, self._x2, self._z3 = self._mat2euler(self._M)

    def __init__(self, arg1=None, x2=None, z3=None):
        if arg1 is None:
            self._init_from_angles(0, 0, 0)  # loads identity matrix
        elif x2 is not None:
            self._init_from_angles(arg1, x2, z3)
        elif arg1.size == 3:
            self._init_from_angles(arg1[0], arg1[1], arg1[2])
        else:
            self._init_from_matrix(arg1)

    def matrix(self, new_matrix=None):
        if new_matrix is not None:
            self._init_from_matrix(new_matrix)
        return self._M

    def euler_angles(self, z1=None, x2=None, z3=None):
        if z1 is not None:
            self._init_from_angles(z1, x2, z3)
        return (self._z1, self._x2, self._z3)

    def random(self):
        V = 2. * math.pi * np.random.random(), np.arccos(
            2.0 * np.random.random() - 1.0), 2. * math.pi * np.random.random()
        self.euler_angles(V)


class TripletHamiltonian:
    def __init__(self):
        self.Id = np.matrix('1 0 0; 0 1 0; 0 0 1', dtype=np.complex_)
        self.Sz = np.matrix('1 0 0; 0 0 0; 0 0 -1', dtype=np.complex_)
        self.Sx = np.matrix('0 1 0; 1 0 1; 0 1 0', dtype=np.complex_) / math.sqrt(2.0)
        self.Sy = - 1j * np.matrix('0 1 0; -1 0 1; 0 -1 0', dtype=np.complex_) / math.sqrt(2.0)

    def fine_structure(self, D, E, rotation=Rotation()):
        rotation_matrix = rotation.matrix()
        rSx = rotation_matrix[0, 0] * self.Sx + rotation_matrix[0, 1] * self.Sy + rotation_matrix[0, 2] * self.Sz
        rSy = rotation_matrix[1, 0] * self.Sx + rotation_matrix[1, 1] * self.Sy + rotation_matrix[1, 2] * self.Sz
        rSz = rotation_matrix[2, 0] * self.Sx + rotation_matrix[2, 1] * self.Sy + rotation_matrix[2, 2] * self.Sz
        return D * (np.dot(rSz, rSz) - 2. * self.Id / 3.) + E * (np.dot(rSy, rSy) - np.dot(rSx, rSx))

    def zeeman(self, Bx, By, Bz):
        return Bx * self.Sx + By * self.Sy + Bz * self.Sz

    def spin_hamiltonian_mol_basis(self, D, E, B, theta, phi):
        Bz = B * math.cos(theta)
        Bx = B * math.sin(theta) * math.cos(phi)
        By = B * math.sin(theta) * math.sin(phi)

        return self.fine_structure(D, E) + self.zeeman(Bx, By, Bz)

    def spin_hamiltonian_field_basis(self, D, E, B, theta, phi):
        return self.fine_structure(D, E, Rotation(0, -theta, -phi + math.pi / 2.)) + self.zeeman(0, 0, B)

    def eval(self, D, E, B, theta=0, phi=0, mol_basis=False):
        if mol_basis:
            return np.linalg.eigvalsh(self.spin_hamiltonian_mol_basis(D, E, B, theta, phi))
        else:
            return np.linalg.eigvalsh(self.spin_hamiltonian_field_basis(D, E, B, theta, phi))


class TwoTriplets:
    def __init__(self):
        self.triplet = TripletHamiltonian()
        self.E = None
        self.D = None
        self.J = None
        self.Jdip = None
        self.B = None
        self.matrix_size = 9

        s2i3 = math.sqrt(2.0 / 3.0);
        si2 = 1.0 / math.sqrt(2.0);
        si3 = 1.0 / math.sqrt(3.0);
        si6 = 1.0 / math.sqrt(6.0);

        self.Jproj = np.array([[0, 0, si3, 0, -si3, 0, si3, 0, 0],
                               [0, 0, 0, 0, 0, -si2, 0, si2, 0],
                               [0, 0, -si2, 0, 0, 0, si2, 0, 0],
                               [0, -si2, 0, si2, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 1.0],
                               [0, 0, 0, 0, 0, si2, 0, si2, 0],
                               [0, 0, si6, 0, s2i3, 0, si6, 0, 0],
                               [0, si2, 0, si2, 0, 0, 0, 0, 0],
                               [1.0, 0, 0, 0, 0, 0, 0, 0, 0]])

    def exchange_matrix(self):
        return np.kron(self.triplet.Sx, self.triplet.Sx) + np.kron(self.triplet.Sy, self.triplet.Sy) + np.kron(
            self.triplet.Sz, self.triplet.Sz)

    def dipole_dipole_matrix(self, uvec):
        """
        returns dipole-dipole interaction matrix, assumes that uvec an unit normalized 3d vector
        """
        unorm = np.linalg.norm(uvec)
        uvec = uvec / unorm
        uS = uvec[0] * self.triplet.Sx + uvec[1] * self.triplet.Sy + uvec[2] * self.triplet.Sz
        return (self.exchange_matrix() - 3. * np.kron(uS, uS))

    def load_field_basis_Hamiltonian(self, triplet1_rotation, triplet2_rotation, dip_vec=None):
        H1 = self.triplet.fine_structure(self.D, self.E, triplet1_rotation) + self.triplet.zeeman(0, 0, self.B)
        H2 = self.triplet.fine_structure(self.D, self.E, triplet2_rotation) + self.triplet.zeeman(0, 0, self.B)
        self.Hfull = np.kron(H1, self.triplet.Id) + np.kron(self.triplet.Id, H2) + self.J * self.exchange_matrix()
        if dip_vec is not None:
            self.Hfull += self.Jdip * self.dipole_dipole_matrix(dip_vec)

    def diag(self):
        self.eval, self.evec = np.linalg.eigh(self.Hfull)

    def quintet_content(self, i):
        iProj = np.dot(self.Jproj[4:9, 0:9], self.evec[0:9, i:i + 1]);
        norm2 = np.dot(np.matrix.getH(iProj), iProj);
        return norm2[0, 0].real;

    def triplet_content(self, i):
        iProj = np.dot(self.Jproj[1:4, 0:9], self.evec[0:9, i:i + 1]);
        norm2 = np.dot(np.matrix.getH(iProj), iProj);
        return norm2[0, 0].real;

    def singlet_content(self, i):
        iProj = np.dot(self.Jproj[0:1, 0:9], self.evec[0:9, i:i + 1]);
        norm2 = np.dot(np.matrix.getH(iProj), iProj);
        return norm2[0, 0].real;

    def sz_elem(self, i):
        Sz2 = np.kron(self.triplet.Sz, self.triplet.Id) + np.kron(self.triplet.Id, self.triplet.Sz)
        vi = self.evec[:, i]
        Sz2ii = reduce(np.dot, [np.matrix.getH(vi), Sz2, vi])
        return Sz2ii[0, 0].real

    def singlet_projector(self):
        singlet_state = np.asmatrix(self.Jproj[0:1, :])
        return np.dot(np.matrix.getH(singlet_state), singlet_state)

    def Bac_field_basis_matrix(self):
        return np.kron(self.triplet.Sx, self.triplet.Id) + np.kron(self.triplet.Id, self.triplet.Sx)

    def print_info(self):
        print("# D %g" % self.D)
        print("# E %g" % self.E)
        print("# B %g" % self.B)
        print("# J %g" % self.J)
        print("# Jip %g" % self.Jdip)


class ODMR_Signal:
    """ 
    * ODMR_Signal
    *
    * Output : Computes ODMR and magnetic resonance signals 
    *
    * Input : spins, a reference on SpinSystem object
    * SpinSystem should define 
    * spins.matrix_size
    * spins.evec
    * spins.eval
    * spins.singlet_projector()
    * spins.Bac_field_basis_matrix()
    """

    def __init__(self, spin_system):
        self.spins = spin_system
        self.rho0 = np.empty(self.spins.matrix_size, dtype=float)
        self.rho2 = np.empty([self.spins.matrix_size, self.spins.matrix_size], dtype=np.complex_)
        self.gamma = None
        self.gamma_diag = None

    def update_from_spin_hamiltonian(self):
        self.Sproj_eig_basis = reduce(np.dot, [np.matrix.getH(self.spins.evec), self.spins.singlet_projector(),
                                               self.spins.evec])
        self.V = reduce(np.dot, [np.matrix.getH(self.spins.evec), self.spins.Bac_field_basis_matrix(), self.spins.evec])

    def omega_nm(self, n, m):
        return self.spins.eval[n] - self.spins.eval[m]

    def load_rho0_thermal(self, Temp):
        sum = 0
        for i in range(self.spins.matrix_size):
            rho0_i = math.exp(- self.spins.eval[i] / Temp)
            self.rho0[i] = rho_i
            sum += rho_i
        self.rho0 /= sum

    def load_rho0_from_singlet(self):
        sum = 0
        for i in range(self.spins.matrix_size):
            self.rho0[i] = self.Sproj_eig_basis[i, i].real
            sum += self.rho0[i]
        self.rho0 /= sum

    def chi1(self, omega):
        c1 = 0j
        for m in range(self.spins.matrix_size):
            for n in range(self.spins.matrix_size):
                # the contribution to chi1 vanishes for n == m, whether gamma is the same for diagonal and non diagonal elements is not relvant here
                Vmn = self.V[m, n]
                Vmn_abs2 = Vmn.real * Vmn.real + Vmn.imag * Vmn.imag
                c1 -= (self.rho0[m] - self.rho0[n]) * Vmn_abs2 / (self.omega_nm(n, m) - omega - 1j * self.gamma);
        return c1

    def find_rho2_explicit(self, omega):
        for m in range(self.spins.matrix_size):
            for n in range(self.spins.matrix_size):
                rrr = 0j
                for nu in range(self.spins.matrix_size):
                    for p in [-1., 1.]:
                        gamma_nm = self.gamma_diag if m == n else self.gamma
                        rrr += (self.rho0[m] - self.rho0[nu]) * self.V[n, nu] * self.V[nu, m] / (
                        (self.omega_nm(n, m) - 1j * gamma_nm) * (self.omega_nm(nu, m) - omega * p - 1j * self.gamma))
                        rrr -= (self.rho0[nu] - self.rho0[n]) * self.V[n, nu] * self.V[nu, m] / (
                        (self.omega_nm(n, m) - 1j * gamma_nm) * (self.omega_nm(n, nu) - omega * p - 1j * self.gamma))
                self.rho2[n, m] = rrr

    def find_rho2(self, omega):
        Vtmp = np.zeros((self.spins.matrix_size, self.spins.matrix_size), dtype=np.complex_)
        for m in range(self.spins.matrix_size):
            for nu in range(self.spins.matrix_size):
                for p in [-1., 1.]:
                    Vtmp[nu, m] += (self.rho0[m] - self.rho0[nu]) * self.V[nu, m] / (
                    self.omega_nm(nu, m) - omega * p - 1j * self.gamma)
        self.rho2 = np.dot(self.V, Vtmp) - np.dot(Vtmp, self.V)
        for m in range(self.spins.matrix_size):
            for n in range(self.spins.matrix_size):
                gamma_nm = self.gamma_diag if m == n else self.gamma
                self.rho2[n, m] /= (self.omega_nm(n, m) - 1j * gamma_nm);

    def odmr(self, omega):
        odmr_amp = 0j
        self.find_rho2(omega)

        for m in range(self.spins.matrix_size):
            for n in range(self.spins.matrix_size):
                odmr_amp += self.rho2[m, n] * self.Sproj_eig_basis[n, m]

        return odmr_amp.real


################################################
dataDC2 = np.loadtxt("testupto30up.txt", comments='%')  # , usecols=(0,1,3),unpack=True)
fieldDC2 = np.zeros(29)
freqDC2 = (dataDC2[650:1415, 0]) / 1e6
freqStartDC2 = freqDC2[0]
NumPoints = 765
freqStopDC2 = freqDC2[764]
freqStepDC2 = freqDC2[11] - freqDC2[10]
IntensityDC2 = np.zeros((29, 765))

# http://python3porting.com/differences.html#range-and-xrange
for i in xrange(29):
    fieldDC2[i] = np.mean(dataDC2[i * 5000:(i + 1) * 5000, 1])
    IntensityDC2[i, :] = dataDC2[i * 5000 + 650:i * 5000 + 1415, 3]

dA = 5.0  # 45
a = math.radians(90.0) * (1.0 / dA + 1.0)  # 91 degree for theta and phi
b = a / dA  # 45 #step for angles

# http://stackoverflow.com/a/2958717/1032286
c = 81.0 / 28.0  # 30 #field step
d = 80.0 + c  # field limit
tau = 5.0

# angles and field
Phi = np.arange(0, a, b)
Theta = np.arange(0, a, b)
Magnetic = np.arange(0, d, c)
Phi_deg = np.zeros(len(Phi))
Theta_deg = np.zeros(len(Theta))
print len(Phi), len(Theta)

Na = len(Phi) * len(Theta)
Np = IntensityDC2.size
Nb = len(fieldDC2)

LambdaM = np.zeros((Np, Na))
LambdaMepr = np.zeros((Np, Na))

triplet_pair = TwoTriplets()
triplet_pair.D = 487.9
triplet_pair.E = 72.9
odmr = ODMR_Signal(triplet_pair)
# for B: 2.9 mT = 81.27236559069694 MHz
# 19.9 mT = 557.7 MHz
# 12 mT = 336.3 MHz

index_Phi = 0
index_a = 0
V1 = 2. * math.pi * np.random.rand(3)
V2 = 2. * math.pi * np.random.rand(3)
Ur = random_unit_vector()
triplet_pair.load_field_basis_Hamiltonian( Rotation(V1), Rotation(V2), Ur )
triplet_pair.diag()
si = 0
quintet_max = 0
for i in range(0,9):
    si += math.pow(triplet_pair.quintet_content(i), 4.0)

if si > quintet_max:
    quintet_max = si
    quintet_angles1 = V1
    quintet_angles2 = V2
    quintet_rdip = Ur
"""
trp.evecs()
odmr.update_from_spin_hamiltonian()
odmr_from_triplets.update_from_spin_hamiltonian()
odmr.load_rho0_from_singlet()
odmr.gamma = 1e-2
odmr.gamma_diag = 1e-2
"""

for triplet_pair.phi in Phi:
    index_Theta = 0
    Phi_deg[index_Phi] = round(math.degrees(Phi[index_Phi]))
    for triplet_pair.theta in Theta:
        index_B = 0
        index_p = 0
        # print(index_a)
        Theta_deg[index_Theta] = round(math.degrees(Theta[index_Theta]))
        for i in xrange(len(freqDC2)):
            for triplet_pair.B in Magnetic:
#                trp.evecs(trp.D, trp.E, trp.B, trp.theta, trp.phi)
                odmr.gamma = 1e-2
                odmr.gamma_diag = 1e-2
#                odmr.doV()
                odmr.Temp = 41600000000
                triplet_pair.load_field_basis_Hamiltonian( Rotation(quintet_angles1), Rotation(quintet_angles2), quintet_rdip )
                triplet_pair.diag()
                odmr.load_rho0_thermal(odmr.Temp)
#                vals = sorted(trp.evals(trp.D, trp.E, trp.B, trp.theta, trp.phi, mol_basis=True))
#                x1 = (vals[1].real - vals[0].real)
#                x2 = (vals[2].real - vals[0].real)
#                LambdaM[index_p][index_a] = ((1.0 / (math.pow(((freqDC2[i] - x1)/ tau), 2.0) + 1.0)) + (1.0 / (math.pow(((freqDC2[i] - x2) / tau), 2.0) + 1.0))) * math.sin(trp.theta)
                LambdaMepr[index_p][index_a] = odmr.chi1(2*math.pi*freqDC2[i])
                index_p += 1
                index_B += 1
        index_a += 1
        index_Theta += 1
    index_Phi += 1

LamInv = np.linalg.pinv(LambdaM)
Experiment = IntensityDC2.flat
pVec1 = np.dot(LamInv, Experiment)

# read weights from a file
pMatrix = np.reshape(pVec1, (len(Phi), len(Theta)))
TheoryVec = np.dot(LambdaM, pVec1)
TheoryMatr = np.reshape(TheoryVec, (765, 29))

pVec2, rnorm1 = nnls(LambdaM,Experiment)
pMatrix2 = np.reshape(pVec2, (len(Phi), len(Theta)))
TheoryVec2 = np.dot(LambdaM, pVec2)
TheoryMatr2 = np.reshape(TheoryVec2, (765, 29))

pVec3, rnorm2 = nnls(LambdaMepr,Experiment)
pMatrix3 = np.reshape(pVec3, (len(Phi), len(Theta)))
TheoryVec3 = np.dot(LambdaMepr, pVec3)
TheoryMatr3 = np.reshape(TheoryVec3, (765, 29))

gnufile = open('TwoTrpTheor5nnlsEPR.dat', 'w+')
for i in xrange(765):
    for j in xrange(29):
        gnufile.write(str(freqDC2[i]) + '  ' + str(fieldDC2[j])+ '  ' + str(TheoryMatr[i][j]) + '  ' + str(TheoryMatr2[i][j]) + '  ' + str(TheoryMatr3[i][j]) + '\n')
    gnufile.write("\n")
gnufile.clos