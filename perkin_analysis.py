# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 21:26:52 2016
Perkin Elmer integrated data read and plot the peak center, amplitude and area
@author: Karthick Perumal
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import re
from PIL import Image
from lmfit import Model
from lmfit.models import LorentzianModel, LinearModel, GaussianModel, ExponentialModel, PseudoVoigtModel, VoigtModel
import glob2
import time
from scipy import sparse

###############################################################################
start_time = time.time()
###############################################################################

def readfolder(folder_range):
    peak1pos1, peak1height1, peak2pos1, peak2height1 = ([] for i in range (4))
    for folder in range(folder_range[0], folder_range[1]+1):
        foldertoread = foldername + str(folder)+"_perkin"
        os.chdir('/gpfs/current/raw/'+foldertoread)
        peak1pos, peak1height, peak2pos, peak2height = ([] for i in range (4))
        for avg_file in sorted(glob2.glob('*.avg')):
            data = readavg(avg_file)  
            #print(avg_file)        
            pv1_pos, pv1_height, lor2_pos, lor2_height = fitsample(data, theta_initial, theta_final)
            
            #print (pv1_pos, pv1_height, lor2_pos, lor2_height)
            peak1pos.append(pv1_pos)
            peak1height.append(pv1_height)
            peak2pos.append(lor2_pos)
            peak2height.append(lor2_height)
        peak1pos1.append(peak1pos)
        peak1height1.append(peak1height)
        peak2pos1.append(peak2pos)
        peak2height1.append(peak2height)
        print("folder " + foldertoread +  " executed in " +str( (time.time() - start_time)) + " seconds..." )
    peak1pos_matrix = np.array(peak1pos1)
    peak1height_matrix = np.array(peak1height1)
    peak2pos_matrix = np.array(peak2pos1)
    peak2height_matrix = np.array(peak2height1)
    fig = plt.figure(1)
    fig.suptitle(foldername + str(folder_range[0])+ "-"+ str(folder_range[1]))
    ax1 = plt.subplot(221)
    plt.imshow(peak1pos_matrix, cmap='jet',  interpolation= 'none', aspect ='auto')
    plt.colorbar()
    ax2 = plt.subplot(222)
    plt.imshow(peak1height_matrix, cmap='jet',  interpolation= 'none', aspect ='auto')
    plt.colorbar()
    ax3 = plt.subplot(223)
    plt.imshow(peak2pos_matrix, cmap='jet',  interpolation= 'none', aspect ='auto')
    plt.colorbar()
    ax4 = plt.subplot(224)
    plt.imshow(peak2height_matrix, cmap='jet',  interpolation= 'none', aspect ='auto')
    plt.colorbar()
    ax1.set_title('peak 1 position ')
    ax2.set_title('peak 1 height')
    ax3.set_title('peak 2 position ')
    ax4.set_title('peak 2 height')
    print("Program Executed in " +str( (time.time() - start_time)) + " seconds..." )
    plt.show()
    return
        

###############################################################################

###############################################################################

def readavg(filename):
    """
    READ *.avg file into dictionary of arrays
    """
    f = open(filename)    
    data = np.genfromtxt(filename,comments = '#', delimiter = "", dtype = None)
    return data
 
###############################################################################

###############################################################################

def gaussian(x, amp, cen, wid):
    """FWHM is approx. 2.354*wid"""
    return(amp/(np.sqrt(2*np.pi)*wid)) * np.exp(-(x-cen)**2 /(2*wid**2))

def line(x, slope, intercept):
    "line"
    return (slope*x + intercept)

def lorentian(x, amp, cen, wid):
    return (amp/np.pi)*(wid/((x-cen)**2 + wid**2))

###############################################################################

###############################################################################


def fitsample(data, theta_initial, theta_final):        
    
    
    x = data[:,0]
    y = data[:,1]
    m = (x > theta_initial) & (x < theta_final)
    x_fit = x[m]
    y_fit = y[m]

    

    pseudovoigt1 = VoigtModel(prefix = 'pv1_')    
    pars= pseudovoigt1.make_params()
    pars['pv1_center'].set(13.5, min = 13.4, max = 13.6)
    pars['pv1_sigma'].set(0.05, min= 0.01, max = 0.1)
    pars['pv1_amplitude'].set(70, min = 1, max = 100)
    #pars['pv1_fraction'].set(0.5)
    

    lorentz2 = LorentzianModel(prefix = 'lor2_')
    pars.update(lorentz2.make_params())
    pars['lor2_center'].set(13.60, min = 13.4, max = 13.9)
    pars['lor2_sigma'].set(0.1, min= 0.01)
    pars['lor2_amplitude'].set(10, min = 1, max = 50 )
    #pars['lor2_fraction'].set(0.5)
    
    line1 = LinearModel(prefix ='l1_')
    pars.update(line1.make_params())
    pars['l1_slope'].set(0)
    pars['l1_intercept'].set(240, min = 200, max = 280)

    
    
    mod = pseudovoigt1 + lorentz2 + line1
    v = pars.valuesdict()
     
    result = mod.fit(y_fit, pars, x=x_fit)    

    #print(result.fit_report())    
    pv1_pos = result.params['pv1_center'].value
    pv1_height = result.params['pv1_height'].value
    lor2_pos = result.params['lor2_center'].value
    lor2_height = result.params['lor2_height'].value
    #peak_area = pars['gau1_fwhm'].value*peak_amp
    #plt.xlim([theta_initial, theta_final])
    #plt.ylim([100, 500])
    #plt.semilogy(x_fit, y_fit, 'bo')
    
    #plt.semilogy (x_fit, result.init_fit, 'k--')    
    #plt.semilogy(x_fit, result.best_fit, 'r-')
    #plt.show()
    return pv1_pos, pv1_height, lor2_pos, lor2_height


###############################################################################

###############################################################################



###############################################################################

###############################################################################


os.chdir('/gpfs/current/raw/')
foldername = 'p08_128_GST-Si111_00'
theta_initial = 13.2
theta_final = 14
folder_range = [370, 380]
readfolder(folder_range)

print("Program Executed in " +str( (time.time() - start_time)) + " seconds..." )

