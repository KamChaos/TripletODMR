0.0 0.0 0.0 415.0 560.8
0.0 0.0 2.89285714286 414.942624619 560.857375381
0.0 0.0 5.78571428571 414.770768568 561.029231432
0.0 0.0 8.67857142857 414.485235803 561.314764197
0.0 0.0 11.5714285714 414.087345535 561.712654465
0.0 0.0 14.4642857143 413.578902314 562.221097686
0.0 0.0 17.3571428571 412.962156368 562.837843632
0.0 0.0 20.25 412.239756146 563.560243854
0.0 0.0 23.1428571429 411.414695289 564.385304711
0.0 0.0 26.0357142857 410.490256309 565.309743691
0.0 0.0 28.9285714286 409.469953175 566.330046825
0.0 0.0 31.8214285714 408.357474793 567.442525207
0.0 0.0 34.7142857143 407.15663103 568.64336897
0.0 0.0 37.6071428571 405.87130262 569.92869738
0.0 0.0 40.5 404.505395858 571.294604142
0.0 0.0 43.3928571429 403.062802669 572.737197331
0.0 0.0 46.2857142857 401.5473663 574.2526337
0.0 0.0 49.1785714286 399.962852629 575.837147371
0.0 0.0 52.0714285714 398.312926862 577.487073138
0.0 0.0 54.9642857143 396.601135253 579.198864747
0.0 0.0 57.8571428571 394.830891379 580.969108621
0.0 0.0 60.75 393.005466438 582.794533562
0.0 0.0 63.6428571429 391.127983046 584.672016954
0.0 0.0 66.5357142857 389.201411988 586.598588012
0.0 0.0 69.4285714286 387.228571429 588.571428571
0.0 0.0 72.3214285714 385.212128123 590.587871877
0.0 0.0 75.2142857143 383.154600218 592.645399782
0.0 0.0 78.1071428571 381.058361276 594.741638724
0.0 0.0 81.0 3
import numpy as np
from numpy.linalg import eig
import cmath
import math
import sys
from functools import reduce
import pandas as pd
import pylab as pl
import array
from scipy.fftpack import fft, ifft

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
        return self.fine_structure(D, E, Rotatino(0, -theta, -phi + math.pi / 2.)) + self.zeeman(0, 0, B)

    def eval(self, D, E, B, theta=0, phi=0, mol_basis=True):
        if mol_basis:
            return np.linalg.eigvalsh(self.spin_hamiltonian_mol_basis(D, E, B, theta, phi))
        else:
            return np.linalg.eigvalsh(self.spin_hamiltonian_field_basis(D, E, B, theta, phi))

a = 91*math.pi/180 #91 градус как предел для фи и тета
b = a/5#45 #шаг для фи и тета
c = 81/28#30 #шаг для поля
d = 80+c #предел для поля

#сами углы и поле
Phi = np.arange(0,a,b)
Theta = np.arange(0,a,b)
Magnetic = np.arange(0,d,c)

#w = np.zeros(len(Phi)*len(Theta))
#weights = pd.DataFrame(np.zeros(len(Phi),len(Theta)), index=Phi, columns=Theta)

trp = TripletHamiltonian()
trp.D = 487.9
trp.E = 72.9
f = open('checkCycle1.txt', 'w')
koef = 10^6 #MHz
#для магнитного поля Бэ: 2.9 мТл = 81.27236559069694 МГц
index = 0
for trp.phi in Phi:
    for trp.theta in Theta:
        for trp.B in Magnetic:
            val1 = trp.eval(trp.D, trp.E, trp.B, trp.theta, trp.phi, mol_basis=True)
            x1 = val1[1] - val1[0]
            x2 = val1[2] - val1[0]
 #           weights[Phi,Theta] = x1
#            w += Intensity[trp.B,x1*koef]
#            w += Intensity[trp.B,x2*koef]
            print(trp.phi,trp.theta,trp.B,x1,x2, file = f)


f.close
print ('Ya konchil')