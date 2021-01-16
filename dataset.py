import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from preprocess import * 
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import dlib

class LRWDataset_LipNet(Dataset):
    
    def __init__(self,data_dir,mode, feature_method, vocab2int=None): 
       
        self.mode = mode
        self.feature_method = feature_method
        self.data_dir = data_dir

        self.data_files = [] 
        
        # vocab dictionary to transform word to integer
        if vocab2int is None: 
            self.vocab2int = {}
        else: 
            self.vocab2int = vocab2int

        # Data is organized: data_dir > WORD > MODE > WORD_ID.txt
        count = 0
        for word in os.listdir(data_dir): 
            if vocab2int is None: 
                self.vocab2int[word] = count
                count += 1
            
            for path in glob.glob(os.path.join(data_dir,word,mode,'*.txt')): 
                fname = os.path.splitext(os.path.basename(path))[0]
                self.data_files.append(fname) 
        
        # initialize detector and predictor for dlib 
        #if feature_method == 'dlib': 
        #    self.detector = dlib.get_frontal_face_detector()
        #    self.predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

    def __len__(self): 
        return len(self.data_files)

    def __getitem__(self, indx): 

        fname = self.data_files[indx]
        
        # load label 
        label_vocab = fname.split('_')[0]
        label = self.vocab2int[label_vocab]

        # vid_path = f'{self.data_dir}/{label_vocab}/{self.mode}/{fname}.mp4'
        frames_path = f'{self.data_dir}/{label_vocab}/{self.mode}/{fname}_mouth_frames.npy'

        # TO DO: load facial features 
        if self.feature_method == 'dlib': 
            #inputs = dlib_get_frames_mouth_only(vid_path, self.detector, self.predictor, normalize=True)
            inputs = np.load(frames_path)
            inputs = np.transpose(inputs, (3,0,1,2))
           #elif self.feature_method == 'bob': 
        # TO DO: other facial feature extraction methods: 
        # elif self.feature_method == 'ASM': 
        # elif self.feature_method == 'LCACM': 
        else: 
            raise NameError("feature method not defined")

        return inputs, label

class LRWDataset_features(Dataset):
    
    def __init__(self,data_dir,mode, feature_method, vocab2int=None): 
       
        self.mode = mode
        self.feature_method = feature_method
        self.data_dir = data_dir

        self.data_files = [] 
        
        # vocab dictionary to transform word to integer
        if vocab2int is None: 
            self.vocab2int = {}
        else: 
            self.vocab2int = vocab2int

        # Data is organized: data_dir > WORD > MODE > WORD_ID.txt
        count = 0
        for word in os.listdir(data_dir): 
            if vocab2int is None: 
                self.vocab2int[word] = count
                count += 1
            
            for path in glob.glob(os.path.join(data_dir,word,mode,'*.txt')): 
                fname = os.path.splitext(os.path.basename(path))[0]
                self.data_files.append(fname) 
        
        # initialize detector and predictor for dlib 
        # if feature_method == 'dlib': 
        #    self.detector = dlib.get_frontal_face_detector()
        #    self.predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

    def __len__(self): 
        return len(self.data_files)

    def __getitem__(self, indx): 

        fname = self.data_files[indx]
        
        # load label 
        label_vocab = fname.split('_')[0]
        label = self.vocab2int[label_vocab]

        # vid_path = f'{self.data_dir}/{label_vocab}/{self.mode}/{fname}.mp4'
        features_path = f'{self.data_dir}/{label_vocab}/{self.mode}/{fname}.npy'

        # TO DO: load facial features 
        if self.feature_method == 'dlib': 
           # inputs = dlib_features_from_vid_path(vid_path, self.detector, self.predictor, normalize=True)
           inputs = np.load(features_path)
           inputs = inputs.reshape((inputs.shape[0],inputs.shape[1]*inputs.shape[2]))
        #elif self.feature_method == 'bob': 
        # TO DO: other facial feature extraction methods: 
        # elif self.feature_method == 'ASM': 
        # elif self.feature_method == 'LCACM': 
        else: 
            raise NameError("feature method not defined")

        return inputs, label
        






