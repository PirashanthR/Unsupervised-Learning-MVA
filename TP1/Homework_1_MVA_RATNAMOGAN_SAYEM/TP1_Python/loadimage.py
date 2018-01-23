# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 23:37:59 2017

@Pirashanth & Othmane
"""

import cv2
import numpy as np


def read_necessary_images(path_data_set,individual):
    '''Equivalent to the matlab function given with the tp: allows to read the set of needed images
    Parameter : path_data_set: (str) path to the images folder
                individual: (int) {1,2,3} individu for which we want to get the images
    '''
    list_data = ['A+000E+00','A+000E+45','A+000E-35','A-035E+15','A+035E+15',\
                 'A+020E+10','A-025E+00','A-015E+20','A+035E-20','A+050E+00']
    full_path = path_data_set+ '\yaleB0' +str(individual)+'\yaleB0' +str(individual) +'_P00'
    full_data_images = [(cv2.imread(full_path+l+'.pgm',0)) for l in list_data]
    return full_data_images


