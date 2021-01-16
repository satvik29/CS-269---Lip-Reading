# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 19:34:15 2020

@author: chang
Summary: CLM implementation using ASM fitting method and gradient for patch 
         response. 

         Section 1 trains the model and tests on single images from LFPW.
         Section 2 evaluates the model on the entire dataset (LFPW).

"""
"""
This file has depndencies on menpo, menpofit, menpodetect and dlib.

The dataset: LFPW
"""
#%% Section 1
"""
Trains model and evaluates performance for singl images from LFPW
"""

#imports from menpo
import menpo.io as mio
from menpo.visualize import print_progress
from menpofit.aam import PatchAAM
from menpofit.aam import LucasKanadeAAMFitter, WibergInverseCompositional
from menpo.feature import gradient, no_op
from menpo.feature.optional import fast_dsift
from menpofit.clm import CLM, GradientDescentCLMFitter, ActiveShapeModel
from menpofit.clm.expert.ensemble import CorrelationFilterExpertEnsemble
from menpofit.modelinstance import OrthoPDM
from menpodetect import load_dlib_frontal_face_detector

#python imports
from pathlib import Path



#%% Dataset Loader

#load dataset for model training
path_to_images = 'C:/Users/chang/OneDrive/Desktop/269 Deform/lfpw/trainset'

#init training set
training_images = []
for img in print_progress(mio.import_images(path_to_images, verbose=True)):
    #convert to greyscale
    if img.n_channels == 3:
        img = img.as_greyscale()
    #crop to landmarks bounding box with an extra 20% padding
    img = img.crop_to_landmarks_proportion(0.2)
    #rescale image if its diagonal is bigger than 400 pixels
    d = img.diagonal()
    if d > 400:
        img = img.rescale(400.0 / d)
    #append to list
    training_images.append(img)

#%% Model Training

#create CLM model
clm = CLM(training_images, 
          group='PTS', 
          patch_shape=[(15, 15), (23, 23)],
          diagonal=150, scales=(0.5, 1.0),
          holistic_features=gradient,
          max_shape_components=20,
          verbose=True)


#%% Fitting Function
#ASM Fitter
fitter = GradientDescentCLMFitter(clm, gd_algorithm_cls = ActiveShapeModel, n_shape = None) 

#%%
#print fitter results
print(fitter)
#%% Test Images

path_to_images = 'C:/Users/chang/OneDrive/Desktop/269 Deform/lfpw/testset'

path_to_lfpw = Path(path_to_images)

image = mio.import_image(path_to_lfpw / 'image_0030.png') #choose an img to test
image = image.as_greyscale()



#%% Face Tracker

#load detector
detect = load_dlib_frontal_face_detector()

#detect
bboxes = detect(image)
print("{} detected faces.".format(len(bboxes)))

#view
if len(bboxes) > 0:
    image.view_landmarks(group='dlib_0', line_colour='red',
                         render_markers=False, line_width=4);

#%% Shift landmarks

#initial bbox
initial_bbox = bboxes[0]

#fit image
result = fitter.fit_from_bb(image, initial_bbox, max_iters=[15, 5],
                            gt_shape=image.landmarks['PTS'])

#print result
print(result)
#%% View Image
result.view(render_initial_shape=True)
#%% View Results
result.image.view(new_figure=True);
result.final_shape.view();

result.image.view(new_figure=True);
result.initial_shape.view(marker_face_colour='blue');
#%% View Iteration History
result.view_iterations()



#%% Section 2
"""
This sections tests the entire database (LFPW) on the trained model.
"""

import glob 
from menpofit.error.base import euclidean_error
import time
import math

start = time.time()

#load img
path_to_images = r'C:\Users\chang\OneDrive\Desktop\269 Deform\lfpw\trainset'
filelist = glob.glob(path_to_images + "/*.png")

errors = []

for i in range(len(filelist)):
    bboxes = None #reset bboxes
    image = mio.import_image(filelist[i])
    
    #ASM needs images to be greyscale, but throws error if it is already gray scale
    try:
        image = image.as_greyscale()
    except:
        #print('already grayscale: ', i)
        continue
    detect = load_dlib_frontal_face_detector()
    
    #detect
    bboxes = detect(image)
    if len(bboxes) == 0:
        continue

    initial_bbox = bboxes[0]

    #bug in module/library, skips img if boolean axis mismatch.
    try: 
        result = fitter.fit_from_bb(image, initial_bbox, max_iters=[15, 5],
                                gt_shape=image.landmarks['PTS'])
    except:
        #print("error gt_shape.landmarks['PTS']: ", i)
        continue
    
    errors.append(result.final_error(compute_error = euclidean_error)) #append errors

end = time.time()
print(end - start) #display time elapsed
#%%
import numpy as np
errors = [e for e in errors if e is not math.inf]
print(np.mean(errors))
print(np.max(errors), np.min(errors))

print(errors)
