import os
import numpy as np
from PIL import Image
from cv2 import resize
import joblib
import cv2
from vgg16_places_365 import VGG16_Places365


##---------------
# Constants
CONFIDENCE=0.5
##---------------

model_back = VGG16_Places365(weights='places')
#model_back=joblib.load("background_model.joblib")

file_name = 'categories_places365.txt'
classes = []
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])



def detect_back(video_path):
    try:
        cap = cv2.VideoCapture(video_path)

        # Get Frame counts
        frame_count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # Get FPS
        FPS=cap.get(cv2.CAP_PROP_FPS)
        # skip window
        skip_window=int(FPS)  # 1 FPS

        # video len
        vid_len=frame_count/FPS

        # get the len upto multiple of 3
        thresh=int((vid_len//3)*3)

        d_back=[]
        blist=[]
        fcount=0

        while fcount<(thresh):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fcount * skip_window)
            fcount+=1
            # print(fcount,"--------------------------")
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame = cv2.resize(frame, (224,224))
            resized_frame=np.expand_dims(resized_frame, 0)
            preds = model_back.predict(resized_frame)[0]
            conf=max(preds)
          
            top_preds = np.argsort(preds)[::-1][0]
           
            prediction=classes[top_preds]

        #   print(f"Label: {classes[top_preds]}   confidence: {conf}\n")
            if conf>CONFIDENCE:
                blist.append(classes[top_preds])
            else:
                 blist.append("")
            
            # blist.append(classes[top_preds])
            # for every 3 sec reinitialize the list
            print(classes[top_preds],conf)

            if  fcount%3==0:
                d_back.append(blist)
                blist=[]


    except Exception as e:
        print(e)
    finally:
        # Release video capture and writer, and close the text file
        cap.release()

        cv2.destroyAllWindows()
        # print(d_back)
        return d_back