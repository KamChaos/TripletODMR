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
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
plotly.tools.set_credentials_file(username='***', api_key='***')

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

################################################
#ExpData Plot Sam's approach


dataDC2 = np.loadtxt("testupto30up.txt", comments='%')#, usecols=(0,1,3),unpack=True)
fieldDC2 = np.zeros(29)
freqDC2 = (dataDC2[:5000,0])/1e6
freqStartDC2 = freqDC2[0]
NumPoints = 5000
freqStopDC2 = freqDC2[4999]
freqStepDC2 = freqDC2[11]-freqDC2[10]
IntensityDC2 = np.zeros((29,5000))
for i in range(29):
	fieldDC2[i] = np.mean(dataDC2[i*5000:(i+1)*5000,1])
	IntensityDC2[i,:] = dataDC2[i*5000:(i+1)*5000,3]
"""
dataDC2 = np.loadtxt("test123up.txt", comments='%')#, usecols=(0,1,3),unpack=True)
fieldDC2 = np.zeros(120)
freqDC2 = (dataDC2[:2500,0])/1e6
freqStartDC2 = freqDC2[0]
NumPoints = 2500
freqStopDC2 = freqDC2[2499]
freqStepDC2 = freqDC2[11]-freqDC2[10]
IntensityDC2 = np.zeros((120,2500))
for i in range(120):
        fieldDC2[i] = np.mean(dataDC2[i * 2500:(i + 1) * 2500, 1])
        IntensityDC2[i, :] = dataDC2[i * 2500:(i + 1) * 2500, 3]
        print(IntensityDC2[i, :], i)
"""
"""
pl.figure()
pl.pcolor(freqDC2, fieldDC2, IntensityDC2)
pl.xlabel(" Frequency")
pl.ylabel(" B (T)")
pl.show()
"""
#df = pd.read_csv('testupto30_clear.csv')



#вспомогательные чиселки для циклов
a = (90*math.pi/180)*(1/45+1) #91 градус как предел для фи и тета
b = a/45#45 #шаг для фи и тета
c = 81/28#30 #шаг для поля
#c = 336/119#30 #шаг для поля
d = 80+c #предел для поля
#d = 335+c #предел для поля
tau = 5

#сами углы и поле
Phi = np.arange(0,a,b)
Theta = np.arange(0,a,b)
Magnetic = np.arange(0,d,c)
Phi_deg = np.zeros(len(Phi))
Theta_deg = np.zeros(len(Theta))

Na = len(Phi)*len(Theta)
Np = IntensityDC2.size
Nb = len(fieldDC2)

w = np.zeros(Na)
L1 = np.zeros((Np,Na))
L2 = np.zeros((Np,Na))
x1 = np.zeros((Na,Nb))
x2 = np.zeros((Na,Nb))
LambdaM = np.zeros((Np,Na))

trp = TripletHamiltonian()
trp.D = 487.9
trp.E = 72.9
#для магнитного поля Бэ: 2.9 мТл = 81.27236559069694 МГц
#19.9 mT = 557.7 MHz
#12 mT = 336.3 MHz

index_Phi = 0
index_a = 0
for trp.phi in Phi:
    index_Theta = 0
    for trp.theta in Theta:
        index_B = 0
        index_p = 0
        for trp.B in Magnetic:
            # сюда загнать ещё экспериментальное поле?
            x = trp.eval(trp.D, trp.E, trp.B, trp.theta, trp.phi, mol_basis=True)
            x1[index_a,index_B] += (x[1] - x[0])
            x2[index_a,index_B] += (x[2] - x[0])
            index1 = int((x1[index_a,index_B]-freqStartDC2)/freqStepDC2)
            index2 = int((x2[index_a,index_B]-freqStartDC2)/freqStepDC2)
            for i in range(index1-10,index1+10,1):
                if abs(freqDC2[i]-x1[index_a,index_B]) < 2*freqStepDC2:
                    w[index_a] += abs(IntensityDC2[index_B, i-1] + IntensityDC2[index_B, i+1])/2
            for j in range(index2 - 10, index2 + 10, 1):
                if abs(freqDC2[j]-x2[index_a,index_B]) < 2*freqStepDC2:
                    w[index_a] += abs(IntensityDC2[index_B, j-1] + IntensityDC2[index_B, j+1])/2
            for i in range(len(freqDC2)):
                L1[index_p,index_a] += freqDC2[i] - x1[index_a,index_B]
                L2[index_p,index_a] += freqDC2[i] - x2[index_a,index_B]
                print(L1[index_p,index_a], L2[index_p,index_a],index_a )
                LambdaM [index_p,index_a] += (1/(((L1[index_p,index_a]/tau)**2)+1)) + (1/(((L2[index_p,index_a]/tau)**2)+1))
                index_p += 1
            index_B += 1
        index_a += 1
        index_Theta += 1
    index_Phi += 1

wnorm = w/58
LamInv = np.linalg.pinv(LambdaM)
print(wnorm)

"""
дальше нужно минимизировать норму от
LamInv*IntensityDC2.flat = wnorm
"""
