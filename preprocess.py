import numpy as np
from scipy.io import loadmat, savemat
import imutils
from imutils import face_utils
import dlib
import cv2
# import menpo
# import menpofit
import pickle
from pathlib import Path
import gzip
# from menpodetect.dlib.detect import DlibDetector

#bob_model_path = './keypoint_model.pkl.gz'
##load Bob model
#with gzip.open(bob_model_path) as f:
#    bob_model = pickle.load(f)


#def KLT_features_from_vid_path(path, normalize=True): 
#    mat = loadmat(f'{vid_path}.mat')
#    features = np.array(mat['facial_features'])
#    if normalize: 
#        # normalize features according 


#    return features

def dlib_features_from_vid_path(path, dlib_detector, dlib_predictor, normalize=True):
    cap = cv2.VideoCapture(path)
    feature_vecs = []
    rects = []

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 

    frame_ctr = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        features = dlib_features_from_frame(frame, (width,height), dlib_detector, dlib_predictor, normalize=normalize)
        feature_vecs.append(features)
        frame_ctr += 1
    return np.array(feature_vecs,dtype=np.float32)

def dlib_features_from_frame(frame, frame_dims, dlib_detector, dlib_predictor, normalize=True):
    face_rects = dlib_detector(frame, 1)
    # if no faces detected, use whole frame as bounding box
    if len(face_rects) == 0: 
        width, height = frame_dims
        rect = dlib.rectangle(left=0,top=0,right=width,bottom=height)
    else: 
        rect = face_rects[0]

    features = face_utils.shape_to_np(dlib_predictor(frame,rect))
    if normalize:
        # normalize features according to mark 31 (tip of the nose)
        features = features - features[30]
        feature_range = (features.max(axis=0) + np.abs(features.min(axis=0)))/2.0
        features = features / feature_range
    
    return features

def dlib_get_frames_mouth_only(path, dlib_detector, dlib_predictor, normalize=True):
    """
    Implemetnation from https://github.com/rizkiarm/LipNet/ with modifications
    """
    MOUTH_WIDTH = 100
    MOUTH_HEIGHT = 50
    HORIZONTAL_PAD = 0.19
    normalize_ratio = None
    
    mouth_frames = []
    
    cap = cv2.VideoCapture(path)
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        mouth_crop_image = np.zeros((MOUTH_HEIGHT,MOUTH_WIDTH,3), dtype=np.uint8)

        face_rects = dlib_detector(frame, 1)
        if len(face_rects) == 0: 
            rect = dlib.rectangle(left=0,top=0,right=frame_width,bottom=frame_height)
        else: 
            rect = face_rects[0]
        
        points = face_utils.shape_to_np(dlib_predictor(frame,rect))     
        
        if len(points) == 0: 
            mouth_frames.append(mouth_crop_image)
            continue

        mouth_points = points[48:]
        
        mouth_centroid = np.mean(mouth_points[:, -2:], axis=0)

        if normalize_ratio is None:
            mouth_left = np.min(mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
            mouth_right = np.max(mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)

            normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)

        new_img_shape = (int(frame.shape[0] * normalize_ratio), int(frame.shape[1] * normalize_ratio))
        resized_img = cv2.resize(frame, new_img_shape)

        mouth_centroid_norm = mouth_centroid * normalize_ratio

        mouth_l = int(mouth_centroid_norm[0] - MOUTH_WIDTH / 2)
        mouth_r = int(mouth_centroid_norm[0] + MOUTH_WIDTH / 2)
        mouth_t = int(mouth_centroid_norm[1] - MOUTH_HEIGHT / 2)
        mouth_b = int(mouth_centroid_norm[1] + MOUTH_HEIGHT / 2)  
        
        temp_mouth_crop_image = resized_img[mouth_t:mouth_b, mouth_l:mouth_r]
       
        if temp_mouth_crop_image.shape[0] > MOUTH_HEIGHT:
           temp_mouth_crop_image = temp_mouth_crop_image[:MOUTH_HEIGHT,:]
        if temp_mouth_crop_image.shape[1] > MOUTH_WIDTH:
            temp_mouth_crop_image = temp_mouth_crop_image[:,:MOUTH_WIDTH]
            
        mouth_crop_image[:temp_mouth_crop_image.shape[0],:temp_mouth_crop_image.shape[1]] = temp_mouth_crop_image
        
        if normalize: 
            mouth_crop_image = mouth_crop_image.astype(np.float32)/255.0

        mouth_frames.append(mouth_crop_image)
        
    return np.array(mouth_frames)

#def bob_features_from_vid_path(path, ff_detector, model):
#    cap = cv2.VideoCapture(path)
#    feature_vecs = []

#    frame_ctr = 0
#    while cap.isOpened():
#        ret, frame = cap.read()
#        if not ret:
#            break
#        feature_vecs.append(features_from_frame(frame, ff_detector, model))
#    return feature_vecs
    
#def bob_features_from_frame(frame, ff_detector, model):
#    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#    frame_img = menpo.image.Image.init_from_channels_at_back(frame)
#    bboxes = ff_detector(frame_img)
#    fr = model.fit_from_bb(frame_img, bboxes[0], max_iters=100)
#    flat_vec = fr.final_shape.as_vector()
#    #return normalize_vec(flat_vec, fr)
#    return format_vec(flat_vec)

#def bob_format_vec(vec):
#    x_pts, y_pts = vec[::2], vec[1::2]
#    return np.array([[x, y] for x, y in zip(x_pts, y_pts)])
    
## Takes a flat vector and normalizes the x and y axes to fit in 0,1
#def bob_normalize_vec(vec, fr):
#    x_pts, y_pts = vec[::2], vec[1::2]
#    min_bounds, max_bounds = fr.final_shape.bounds()

#    x_range = max_bounds[0] - min_bounds[0]
#    y_range = max_bounds[1] - min_bounds[1]

#    x_norm = (x_pts - min_bounds[0]) / x_range
#    y_norm = (y_pts - min_bounds[1]) / y_range
#    return np.array([[x, y] for x, y in zip(x_norm, y_norm)])

if __name__ == "__main__": 

    data_dir = './data/lipread_20_mp4/'
    modes = ['train', 'val', 'test']
 
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

    for mode in tqdm(modes): 
    
        # Data is organized: data_dir > WORD > MODE > WORD_ID.txt
        print(f'Collecting {mode} filenames...')
        data_files = [] 

        for word in os.listdir(data_dir):  

            for path in glob.glob(os.path.join(data_dir,word,mode,'*.txt')): 
                fname = os.path.splitext(os.path.basename(path))[0]
                data_files.append(fname) 
        
        print('Done collecting filenames!')   
    
        print(f'Starting extracting features from {mode} video...')  
    
        for indx, fname in enumerate(tqdm(data_files)): 

            label_vocab = fname.split('_')[0]

    #         print(fname)

            vid_path = f'{data_dir}/{label_vocab}/{mode}/{fname}.mp4'
    #         arr_save_path = f'{data_dir}/{label_vocab}/{mode}/{fname}.npy'
            arr_save_path = f'{data_dir}/{label_vocab}/{mode}/{fname}_mouth_frames.npy'

            # features = dlib_features_from_vid_path(vid_path, detector, predictor, normalize=True)       
            mouth_frames = dlib_get_frames_mouth_only(vid_path, detector, predictor, normalize=True)
        
    #         np.save(arr_save_path, features)
            np.save(arr_save_path, mouth_frames)