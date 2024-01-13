import torch
import torchvision
import cv2
import os
import numpy as np
import joblib
import torchvision.transforms as tt
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

##--------
# Constants
# Specify the height and width to which each 
# video frame will be resized in our dataset.
IMAGE_HEIGHT , IMAGE_WIDTH = 224, 224

# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 9

# CLASSES_LIST
CLASSES_LIST=['Person Walking','Person Running',"Person Exercising","Person Eating"]
##----------


model_act=joblib.load(r"act_1_joblib.pkl")

# transforms required by r2plus1d model
r21d_trans=tt.Compose([R2Plus1D_18_Weights.KINETICS400_V1.transforms()])

# functions to apply transforms to each frame
def apply_tansforms(feat):

  # list to store transformed frames
  feats=[]
  for i in range(len(feat)):

      #converting to array and reshaping in required format
      x=np.transpose(np.array(feat[i]), (0,3,1,2))
      # convertin to tensor to apply transforms
      a=torch.Tensor(x)
      # apply transforms and append to the list
      feats.append(r21d_trans(a))
  return feats

def val_frames_extraction(video_path,SEQUENCE_LENGTH=9,TIME_SECODNS=3):
    '''
    This function will extract the required frames from a video after resizing and normalizing them.
    Args:
        video_path: The path of the video in the disk, whose frames are to be extracted.
    Returns:
        frames_list: A list containing the resized and normalized frames of the video.
    '''

    # Declare a list to store video frames.

    frames_list = []
    vid_list=[]
    # Read the Video File using the VideoCapture object.
    video_reader = cv2.VideoCapture(video_path)
    # Get Frame counts
    frame_count=video_reader.get(cv2.CAP_PROP_FRAME_COUNT)
    # Get FPS
    FPS=video_reader.get(cv2.CAP_PROP_FPS)
    # Find video length
    vid_len=frame_count/FPS
    # Finding frames in 3 seconds window
    thresh_frames=int(3*FPS)

    # Get the total number of frames in the video.
    # video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the the interval after which frames will be added to the list.
    # as we need 9 frames for each 3 second video
    skip_frames_window = int(thresh_frames/SEQUENCE_LENGTH)

    # Iterate through the Video Frames.
    n_videos=int(vid_len/3)
    print(vid_len/3, f"So dividing into {n_videos} sub videos")
    for i in range(n_videos):
      count=i*SEQUENCE_LENGTH
      for frame_counter in range(SEQUENCE_LENGTH):
          # Set the current frame position of the video.
          video_reader.set(cv2.CAP_PROP_POS_FRAMES, (count+frame_counter) * skip_frames_window)

          # Reading the frame from the video.
          success, frame = video_reader.read()

          # Check if Video frame is not successfully read then break the loop
          if not success:
              break

          # Resize the Frame to fixed height and width.
          resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

          # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
          normalized_frame = resized_frame / 255

          # Append the normalized frame into the frames list
          frames_list.append(normalized_frame)


      vid_list.append(frames_list)
      frames_list =[]

    # Release the VideoCapture object.
    video_reader.release()

    # Return the frames list.
    return vid_list


def preprocess_pred_1(video_file_path):
    # video_file_path = "/content/run_test1.mp4"
    features=[]
    # Extract the frames of the video file.
    vid_list = val_frames_extraction(video_file_path)


    for i in range(len(vid_list)):
      vid_list[i]=np.asarray([vid_list[i]])

      vid_list[i]=apply_tansforms(vid_list[i])[0]

    # Return the list of frames
    return vid_list

def detect_activity(video_path):
    feat_1=preprocess_pred_1(video_path)

    with torch.no_grad():
      model_act.eval()
      y_preds=[]
      y_probas=[]
      for i in feat_1:
        pred=model_act(i.unsqueeze(0))
        # print(pred[0])
        probas=(F.softmax(pred[0],dim=0))
        pred=torch.argmax(pred, dim = 1).to("cpu").numpy()
        y_preds.append(pred)
        y_probas.append(probas)
    print(CLASSES_LIST)
    class_predictions=[CLASSES_LIST[i[0]] for i in y_preds]
    for i,j in zip(y_probas,class_predictions):
      print(i,j)
    #print(class_predictions)
    output=[[i,i,i] for i in class_predictions]
    return output