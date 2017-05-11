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
        self.matrix_size = 3

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
#false? check c++
    def evals(self, D, E, B, theta=0, phi=0, mol_basis=False):
        if mol_basis:
            return np.linalg.eigvals(self.spin_hamiltonian_mol_basis(D, E, B, theta, phi))
        else:
            return np.linalg.eigvals(self.spin_hamiltonian_field_basis(D, E, B, theta, phi))

    def evecs(self, D, E, B, theta=0, phi=0):
        self.eval, self.evec = np.linalg.eigh(self.spin_hamiltonian_field_basis(D, E, B, theta, phi))


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

    def doV(self):
        self.V = reduce(np.dot, [np.matrix.getH(self.spins.evec), self.spins.Sx, self.spins.evec])

    def omega_nm(self, m, n):
        return self.spins.eval[n] - self.spins.eval[m]

    def load_rho0_thermal(self, Temp):
        sum = 0
        for i in range(self.spins.matrix_size):
            rho0_i = math.exp(- self.spins.eval[i] / Temp)
            self.rho0[i] = rho0_i
            sum += rho0_i
        self.rho0 /= sum
        return self.rho0

    def chi1(self, omega):
        c1 = 0j
        for m in range(self.spins.matrix_size):
            for n in range(self.spins.matrix_size):
                # the contribution to chi1 vanishes for n == m, whether gamma is the same for diagonal and non diagonal elements is not relvant here
                Vmn = self.V[m,n]
                Vmn_abs2 = Vmn.real*Vmn.real + Vmn.imag*Vmn.imag
                c1 -= (self.rho0[m] - self.rho0[n]) * Vmn_abs2 / (self.omega_nm(n, m) - omega - 1j * self.gamma);
        return c1

################################################

dataDC2 = np.loadtxt("testupto30up.txt", comments='%')  # , usecols=(0,1,3),unpack=True)
fieldDC2 = np.zeros(29)
freqDC2 = (dataDC2[650:1415, 0]) / 1e6
freqStartDC2 = freqDC2[0]
NumPoints = 765
freqStopDC2 = freqDC2[764]
freqStepDC2 = freqDC2[11] - freqDC2[10]
IntensityDC2 = np.zeros((29, 765))

for i in xrange(29):
    fieldDC2[i] = np.mean(dataDC2[i * 5000:(i + 1) * 5000, 1])
    IntensityDC2[i, :] = dataDC2[i * 5000 + 650:i * 5000 + 1415, 3]

c = 81.0 / 28.0  # 30 #field step
d = 80.0 + c  # field limit


# angles and field
Magnetic = np.arange(0, d, c)

#print len(Phi), len(Theta)
tau = 5
Na = 5#len(Phi) = len(Theta)
Na2 = Na*Na
Np = IntensityDC2.size
Nb = len(fieldDC2)
Phi_deg = np.zeros(Na)
Theta_deg = np.zeros(Na)
LambdaM = np.zeros((Np, Na2))
LambdaMepr = np.zeros((Np, Na2))

trp = TripletHamiltonian()
trp.D = 487.9
trp.E = 72.9
odmr = ODMR_Signal(trp)

index_Phi = 0
index_a = 0

"""
trp.evecs()
odmr.update_from_spin_hamiltonian()
odmr_from_triplets.update_from_spin_hamiltonian()
odmr.load_rho0_from_singlet()
odmr.gamma = 1e-2
odmr.gamma_diag = 1e-2
"""

for ang1 in range(Na):
    trp.phi = (1+math.pi*ang1)/(2*Na)
    index_Theta = 0
    Phi_deg[index_Phi] = round(math.degrees(trp.phi))
    for ang2 in range(Na):
        trp.theta = (1+math.pi*ang2)/(2*Na)
        index_B = 0
        index_p = 0
        Theta_deg[index_Theta] = round(math.degrees(trp.theta))
        print Phi_deg[index_Phi], Theta_deg[index_Theta]
        for i in xrange(len(freqDC2)):
            for trp.B in Magnetic:
                trp.evecs(trp.D, trp.E, trp.B, trp.theta, trp.phi)
                odmr.gamma = 1e-2
                odmr.gamma_diag = 1e-2
                odmr.doV()
                odmr.Temp = 41600000000
                odmr.load_rho0_thermal(odmr.Temp)
                vals = sorted(trp.evals(trp.D, trp.E, trp.B, trp.theta, trp.phi, mol_basis=False))
                x1 = (vals[1].real - vals[0].real)
                x2 = (vals[2].real - vals[0].real)
                LambdaM[index_p][index_a] = ((1.0 / (math.pow(((freqDC2[i] - x1) / tau), 2.0) + 1.0)) + (1.0 / (math.pow(((freqDC2[i] - x2) / tau), 2.0) + 1.0))) * math.sin(trp.theta)
                LambdaMepr[index_p][index_a] = odmr.chi1(2*math.pi*freqDC2[i]).real
                index_p += 1
                index_B += 1
        index_a += 1
        index_Theta += 1
    index_Phi += 1

Experiment = IntensityDC2.flat

pVec1, rnorm1 = nnls(LambdaM,Experiment)
pMatrix1 = np.reshape(pVec1, (len(Phi_deg),len(Phi_deg)))
TheoryVec1 = np.dot(LambdaMepr, pVec1)
TheoryMatr1 = np.reshape(TheoryVec1, (765, 29))

pVec2, rnorm2 = nnls(LambdaMepr,Experiment)
pMatrix2 = np.reshape(pVec2, (len(Phi_deg),len(Phi_deg)))
TheoryVec2 = np.dot(LambdaMepr, pVec2)
TheoryMatr2 = np.reshape(TheoryVec2, (765, 29))

gnufile1 = open('TheoryFromWeights2_5nnls.dat', 'w+')
for i in xrange(765):
    for j in xrange(29):
        gnufile1.write(str(freqDC2[i]) + '  ' + str(fieldDC2[j])  + '  ' + str(TheoryMatr1[i][j]) + '\n')
    gnufile1.write("\n")
gnufile1.close

gnufile2 = open('TheoryFromWeights5nnlsEPR.dat', 'w+')
for i in xrange(765):
    for j in xrange(29):
        gnufile2.write(str(freqDC2[i]) + '  ' + str(fieldDC2[j])  + '  ' + str(TheoryMatr2[i][j]) + '\n')
    gnufile2.write("\n")
gnufile2.close

gnufile3 = open('WeightsAngles5.dat', 'w+')
for i in xrange(Na):
    for j in xrange(Na):
        gnufile3.write(str(Phi_deg[i]) + '  ' + str(Theta_deg[j]) + '  ' + str(pMatrix1[i][j]) + '  ' + str(pMatrix2[i][j]) + '\n')
    gnufile3.write("\n")
gnufile3.close
gnufile4 = open('WeightsAngles5Lambda.dat', 'w+')
for i in xrange(Np):
    for j in xrange(Na2):
        gnufile4.write(str(LambdaM[i][j]) + '  ' + '\n')
    gnufile4.write("\n")
gnufile4.close