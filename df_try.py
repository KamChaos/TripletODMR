import pandas as pd
import numpy as np
import math

#hpi = math.pi/2
a = 91*math.pi/180 #91 градус как предел для фи и тета
b = a/45 #шаг для фи и тета
c = 402/30 #шаг для поля
d = 401+c #предел для поля
Phi = np.arange(0,a,b)
Theta = np.arange(0,a,b)
Magnetic = np.arange(0,d,c)
#creating dataframe for the prospective theoretical calculations
iterables = [Phi, Theta]#, Magnetic]
NumCol = len(Phi)*len(Theta) #number of coloumns in dataframe
NumRow = len(Magnetic) #number of rows in dataframe
index = pd.MultiIndex.from_product(iterables, names=['Phi1', 'Theta1'])#, 'Field'])
Theory = pd.DataFrame(np.random.randn(f,e), index = Magnetic, columns = index) # dataframe
#s = pd.Series(np.random.randn(62775), index=index)

print(daf)
daf.to_excel('attempt_df1.xlsx', sheet_name='attempt_df')
#print(len(Phi),len(Theta),len(Magnetic))
"""
Предыдущий вариант, который работал:
iterables = [Phi, Theta, Magnetic]
index = pd.MultiIndex.from_product(iterables, names=['Phi1', 'Theta1', 'Field'])
s = pd.Series(np.random.randn(62775), index=index)
"""