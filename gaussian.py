import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
 
plt.rcParams['font.size']=14

data_dir = '/home/take/LED_measurements/data/ring/'

def gaussian_func(x, A, mu, sigma):
    return (A * np.exp( - (x - mu)**2 / (2 * sigma**2)))

def cw_data(batch, num):
    data = np.loadtxt(data_dir+'rev2-12_LED5.csv'.format(batch, num), delimiter=',', skiprows=1 )
    
    angle = data[:,2]
    v = data[:,1]
    intensity = (v - 5) / (np.max(v) - 5)
    return angle, intensity

angle, intensity = cw_data(1,1)
arc = np.deg2rad(angle)

x = angle
y = intensity

#plt.xlim(307,322)
#plt.ylim(0, 1)
plt.scatter(x,y)
#plt.show()

parameter_initial = [1, 0.5, 3]
 

popt, pcov = curve_fit(gaussian_func, x, y, parameter_initial)
xd = np.arange(x.min(), x.max(), 0.001)
estimated_curve = gaussian_func(xd, popt[0], popt[1], popt[2])

 

StdE = np.sqrt(np.diag(pcov))
 

alpha=0.025
lwCI = popt + norm.ppf(q=alpha)*StdE
upCI = popt + norm.ppf(q=1-alpha)*StdE
 

mat = np.vstack((popt,StdE, lwCI, upCI)).T
df=pd.DataFrame(mat,index=("A", "mu", "sigma"),
columns=("Estimate", "Std. Error", "lwCI", "upCI"))
print(df, popt, StdE)

lines =[]
#fig, ax = plt.subplots()
plt.plot(xd, estimated_curve, label="190mm \n $\\mu$: {2:.3f}$\\pm${3:.3f} \n  $\\sigma$: {4:.3f}$\\pm${5:.3f} \n ".format(popt[0], StdE[0], popt[1], StdE[1], popt[2], StdE[2]))#, label="Estimated curve "
#ax.legend(lines[:2], [r"Estimate A" %(popt[0]), r"Estimate mu" %(popt[1])], loc='best')




plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=11)
#plt.xlim(307,322)
#plt.ylim(0, 1)
plt.title("LED5")
plt.xlabel("Angle(beta) [Degrees]")
plt.ylabel("Relative intensity")
save_dir = '/home/take/LED_measurements/graph/presen/'
plt.savefig(os.path.join(save_dir, 'Trev2-12-190vs280_LED5vsLED.png'))
plt.show()

