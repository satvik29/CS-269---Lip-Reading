%% PREPROCESS_KLT.M
% Script to generate facial feature points for lip reading task using KLT
% algorithm 

% Requires: Computer Vision Toolbox
clc, clear all, close all

%% 
split_types = {'train', 'val', 'test'}; 

N_FEATURES = 68; 
plotTF = true; 
vid_path = 'C:/Users/KrishKabra/Documents/CS269-LipReading/example_video_AGREE_00001.mp4'; 

%% Collect facial feature points 

% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();

% Read a video frame and run the face detector.
videoReader = VideoReader(vid_path);
bbox = [1 1 videoReader.Width videoReader.Height];

% Video writer 
v = VideoWriter('KLT_example_tracking_video_AGREE_00001.mp4');
open(v)

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
%         videoFrame = insertShape(videoFrame, 'Rectangle', bbox);
        if plotTF
        figure(1), 
            imshow(videoFrame), title('Detected features');
            hold on, 
            plot(points);
            hold off
            frame = getframe(gcf); 
            writeVideo(v,frame);
            for rep=1:3
                writeVideo(v,frame);
            end
%             pause(0.05)
        end
        % initalize tracker
        pointTracker = vision.PointTracker;
        points = points.Location; 
        initialize(pointTracker,points,videoFrame);
    else
        % Track the points 
        [points, isFound] = step(pointTracker, videoFrame);
%                     points = points(isFound, :);
        videoFrame = insertMarker(videoFrame, points, '+', ...
        'Color', 'white');  
        if plotTF
        figure(1), 
            imshow(videoFrame), title('Tracked features');
            frame = getframe(gcf); 
            writeVideo(v,frame);
            for rep=1:3
                writeVideo(v,frame);
            end
%             pause(0.05)
        end
    end              
end
release(pointTracker);

% close video writer
close(v);
 