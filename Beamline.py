# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 15:35:38 2015

@author: perumal
"""
import math
import numpy as np
global h, c
h = 6.626e-34  #joules
c = 2.998e8    #m/sec
eV = 1.602e-19 #Joules


def wavelength(energy):
    #Energy in KeV, so that's why eV is multiplied by 1000 
    #meter has to be converted to Angstroms, so multiplied by 1e10
    wavelength = h*c/(energy*1000*eV)*1e10       
    return wavelength
    
    
def calc_angle(energy, qz):
    #energy = raw_input("Enter energy in KeV   :")
    
        
    theta = math.asin(qz*wavelength(energy)/(4*math.pi))
    return theta*180/math.pi

print (calc_angle(18, 2))

print (wavelength(int(input("Enter energy:"))))

print (math.sin(math.radians(90)))

print (np.sin(np.radians(90)))