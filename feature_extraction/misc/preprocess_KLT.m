%% PREPROCESS_KLT.M
% Script to generate facial feature points for lip reading task using KLT
% algorithm 

% Requires: Computer Vision Toolbox
clc, clear all, close all

%% 

data_dir = 'C:/Users/KrishKabra/Documents/CS269-LipReading/Data/lipread_mp4/'; 
dirExceptions = {'.','..','Preliminary Dataset'};
folders = dir(data_dir); 
words = {folders([folders.isdir]).name}; 
words = words(~ismember(words,dirExceptions)); 

split_types = {'train', 'val', 'test'}; 

% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();

N_FEATURES = 68; 
plotTF = false; % toggle for plotting 

%% loop through words
for i=270:length(words)
    disp(words{i})
    %% loop through split types
    for j=1:length(split_types)
        vid_fnames = {dir([data_dir words{i} '/' split_types{j} '/' '*.mp4']).name}; 
        %% loop through videos 
        for k=1:length(vid_fnames)
            vid_path = [data_dir words{i} '/' split_types{j} '/' vid_fnames{k}]; 
            tic
            %% Collect facial feature points 
            % Read a video frame and run the face detector.
            videoReader = VideoReader(vid_path);
            bbox = [1 1 videoReader.Width videoReader.Height];
            
            facial_features = zeros(videoReader.NumFrames, N_FEATURES, 2); 
            
            FN = 0;
            while hasFrame(videoReader)
                videoFrame = readFrame(videoReader);
                FN = FN+1;
                % Face detection 
                bbox_new = faceDetector(videoFrame); 
                % if only one face found, update bbox as normal
                if size(bbox_new,1)==1
                    bbox = bbox_new;
                % if size(bbox,1)>1, choose biggest bounding box area
                elseif size(bbox_new,1)>1
                    bbox_areas = bbox_new(:,3) .* bbox_new(:,4);
                    [~,ind] = max(bbox_areas); 
                    bbox = bbox_new(ind,:);
                % else size(bbox,1)<1 i.e 0, do not update bounding box        
                end
                
                % VidROI = imcrop(videoFrame, bbox);
                
                % First frame 
                if FN == 1
                    % Initialize facial features to track 
                    points = detectMinEigenFeatures(rgb2gray(videoFrame),'ROI',bbox);
                    points = points.selectStrongest(N_FEATURES); 
                    videoFrame = insertShape(videoFrame, 'Rectangle', bbox);
                    if plotTF
                    figure(1), 
                        imshow(videoFrame), title('Detected features');
                        hold on, 
                        plot(points);
                        hold off
%                         pause(0.1)
                    end
                    % initalize tracker
                    pointTracker = vision.PointTracker;
                    points = points.Location; 
                    initialize(pointTracker,points,videoFrame);
%                     facial_features = zeros(videoReader.NumFrames,size(points,1), 2);
                    facial_features(FN,1:size(points,1),1:size(points,2)) = points; 
                else
                    % Track the points 
                    [points, isFound] = step(pointTracker, videoFrame);
%                     points = points(isFound, :);
                    facial_features(FN,1:size(points,1),1:size(points,2)) = points; 
                    videoFrame = insertMarker(videoFrame, points, '+', ...
                    'Color', 'white');  
                    if plotTF
                    figure(1), 
                        imshow(videoFrame), title('Tracked features');
%                         pause(0.1)
                    end
                end              
            end
            release(pointTracker);
            % save facial features 
            [filepath, name, ext] = fileparts(vid_path); 
            save([filepath '/' name '.mat'], 'facial_features');
            fprintf('Time taken for video: %.3f s \n', toc)
        end
    end
end