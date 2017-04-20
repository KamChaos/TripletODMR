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
dA = 45.0  # 45
a = math.radians(90.0) * (1.0 / dA + 1.0)  # 91 degree for theta and phi
b = a / dA
c = 81/28#30 #шаг для поля
#c = 336/119#30 #шаг для поля
d = 80+c #предел для поля
#d = 335+c #предел для поля

#сами углы и поле
Phi = np.arange(0,a,b)
Theta = np.arange(0,a,b)
Magnetic = np.arange(0,d,c)
Phi_deg = np.zeros(len(Phi))
Theta_deg = np.zeros(len(Theta))
w = np.zeros((len(Phi),len(Theta)))
print(len(Phi),len(Theta))
trp = TripletHamiltonian()
trp.D = 487.9
trp.E = 72.9
#для магнитного поля Бэ: 2.9 мТл = 81.27236559069694 МГц
#19.9 mT = 557.7 MHz
#12 mT = 336.3 MHz

index_Phi = 0
for trp.phi in Phi:
    index_Theta = 0
    Phi_deg[index_Phi] = round(float((Phi[index_Phi] * 180) / math.pi)) #converting radians to degrees
    for trp.theta in Theta:
        Theta_deg[index_Theta] = round(float((Theta[index_Theta] * 180) / math.pi)) #converting radians to degrees
        index_B = 0
        weight_sum = 0
        for trp.B in Magnetic:
            val1 = trp.eval(trp.D, trp.E, trp.B, trp.theta, trp.phi, mol_basis=True)
            x1 = (val1[1] - val1[0])
            x2 = (val1[2] - val1[0])
            index1 = int((x1-freqStartDC2)/freqStepDC2)
            index2 = int((x2-freqStartDC2)/freqStepDC2)
            for i in range(index1-10,index1+10,1):
                if abs(freqDC2[i]-x1) < 2*freqStepDC2:
                    w[index_Phi,index_Theta] += abs(IntensityDC2[index_B,i-1]+IntensityDC2[index_B,i+1])/2

            for j in range(index2 - 10, index2 + 10, 1):
                if abs(freqDC2[j]-x2) < 2*freqStepDC2:
                    w[index_Phi,index_Theta] += abs(IntensityDC2[index_B, j-1] + IntensityDC2[index_B, j + 1])/2
            index_B += 1
        index_Theta += 1
    index_Phi += 1

#w_norm = w/58
w_norm = w/240


"""
np.savetxt("Weights.csv", w, delimiter=",")
np.savetxt("Weights_norm.csv", w_norm, delimiter=",")


weights = pd.DataFrame(w, index=Phi_deg, columns=Theta_deg)
weights_norm = pd.DataFrame(w_norm, index=Phi, columns=Theta)
weights.to_excel('Weights_df.xlsx', sheet_name='attempt_w_df')
weights_norm.to_excel('Weights_df_norm.xlsx', sheet_name='attempt_wn_df')
filecheck = open('checkGlobal.txt', 'w')
print (w,np.amax(w),np.amax(w_norm),file=filecheck)
filecheck.close
attemptx = weights.idxmax()
attempty = weights.idxmax(axis=1)
print(weights.loc[Phi_deg[44],Theta_deg[44]])


#plotting with plotly
TheoryW = [go.Heatmap( z=weights_norm.values.tolist(), colorscale='Viridis')]
py.iplot(TheoryW, filename='pandas-heatmap')
"""

gnufile = open('DC2_weights45.dat','w')

for i in range(len(Phi_deg)):
    for j in range(len(Theta_deg)):
        #print(Phi_deg[i],'  ', Theta_deg[j], '  ', w[i,j]/56, file=gnufile)
        print(Phi_deg[i],'  ', Theta_deg[j], '  ', w[i,j]/240, file=gnufile)
    print("", file=gnufile)
gnufile.close