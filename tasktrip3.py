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

################################################
#ExpData Plot Sam's approach
data = np.loadtxt("testupto30up.txt", comments='%')#, usecols=(0,1,3),unpack=True)
field = np.zeros(29)
freq = data[:5000,0]
freqStart = 1000000
freqStop = 2500000000
freqStep = freq[11]-freq[10]
#print(freq[4999],freqStep)
Intensity = np.zeros((29,5000))
for i in range(29):
	field[i] = np.mean(data[i*5000:(i+1)*5000,1])
	Intensity[i,:] = data[i*5000:(i+1)*5000,3]
#print(Intensity[0,0],Intensity[28,0])
"""
pl.figure()
pl.pcolor(freq, field, Intensity)
pl.xlabel(" Frequency")
pl.ylabel(" B (T)")
pl.show()
"""
#df = pd.read_csv('testupto30_clear.csv')



#вспомогательные чиселки для циклов
a = 91*math.pi/180 #91 градус как предел для фи и тета
b = a/5#45 #шаг для фи и тета
c = 81/28#30 #шаг для поля
d = 80+c #предел для поля

#сами углы и поле
Phi = np.arange(0,a,b)
Theta = np.arange(0,a,b)
Magnetic = np.arange(0,d,c)

w = np.zeros((len(Phi),len(Theta)))
#weights = pd.DataFrame(np.zeros(len(Phi),len(Theta)), index=Phi, columns=Theta)

trp = TripletHamiltonian()
trp.D = 487.9
trp.E = 72.9
koef = 10^6 #MHz
#для магнитного поля Бэ: 2.9 мТл = 81.27236559069694 МГц

#перевод в индексы по частоте: (2500000000-1000000)/5000=freq[2]-freq[1]=freqStep,
index_Phi = 0
#index_Theta = 0
#index_B = 0
for trp.phi in Phi:
    index_Theta = 0
    for trp.theta in Theta:
        index_B = 0
        for trp.B in Magnetic:
            val1 = trp.eval(trp.D, trp.E, trp.B, trp.theta, trp.phi, mol_basis=True)
            x1 = (val1[1] - val1[0])*koef
            x2 = (val1[2] - val1[0])*koef
            index1 = int((x1-freqStart)/freqStep)
            index2 = int((x2-freqStart)/freqStep)
            for i in range(index1-10,index1+10,1):
                if abs(freq[i]-x1) < 20*freqStep:
                    w[index_Phi, index_Theta] += abs(Intensity[index_B,i-1]+Intensity[index_B,i+1])/2
                    #print('w1 =', w)
            for j in range(index2 - 10, index2 + 10, 1):
                if abs(freq[i] - x2) < freqStep:
                    w[index_Phi, index_Theta] += abs(Intensity[index_B, i-1] + Intensity[index_B, i + 1])/2
                    #print('w2 =', w)

#            index +=1
            #print(index_B,w[index_Phi,index_Theta])
            index_B += 1
        index_Theta += 1
    index_Phi += 1

"""
StepByfreq=(2.500000E+9)/5000
w = np.zeros((len(Phi), len(Theta)))
b = df[(df['magnetic'] ==81)]
print(b['frequency'].as_matrix()[int(416*1e6/StepByfreq)])
for trp.phi in range(0,len(Phi)):
    for trp.theta in range(0,len(Theta)):
        for trp.B in Magnetic:
            val1 = trp.eval(trp.D, trp.E, trp.B, Theta[trp.theta], Phi[trp.phi], mol_basis=True)
            x1 = val1[1] - val1[0]
            x2 = val1[2] - val1[0]
            b = df[(df['magnetic'] ==trp.B)]
            if len(b['frequency'].as_matrix())>2:
                w[trp.phi,trp.theta ]+=(b['frequency'].as_matrix()[int(x1*1e5/StepByfreq)])
                #print(w[trp.phi,trp.theta ])

"""
print (np.amax(w))



# получаем набор графиков val2(B) для всех значений theta, phi=0:
#for theta in Theta:
#	mpl.pyplot.plot(Magnetic,[v[0][theta][x][0] for x in Magnetic])#Первый уровень
#	mpl.pyplot.plot(Magnetic,[v[0][theta][x][1] for x in Magnetic])#Второй уровень
#mpl.pyplot.show()
 

 

