import numpy as np
import os.path
import pandas as pd
import math
import matplotlib.pyplot as plt


def CreateClearFile():
    f = open('testupto30_clear.csv', 'w')
    f.write('frequency,magnetic,intensity\n')
    with open('testupto30up.txt', 'r') as fp:
       for line in fp:
           if line[0]!='%':
               b=line.strip().split('\t')
               if len(b)>3:
                   c = [b[0], round(float(b[1]), 4), b[3]]
                   f.write(','.join([str(x) for x in c])+'\n')
       fp.close()
    f.close()

CreateClearFile()

df = pd.read_csv('testupto30_clear.csv')
#print(df)
print('\n b:')
b = df[(df['magnetic'] ==0.0007)]
print(b)

#plt.plot(b['frequency'][1000:1400],b['intensity'][1000:1400])
plt.plot(b['frequency'][:1400],b['intensity'][:1400])
plt.show()
b['frequency'].max,b['intensity']
