import cv2
import os
import joblib
import numpy as np
import torch

## constants
CONFIDENCE=0.4
##


try:
  model_obj=joblib.load("yolov5s.joblib")
except:
  model_obj=torch.hub.load("ultralytics/yolov5","yolov5s",pretrained=True)
#model_obj=torch.hub.load("ultralytics/yolov5","yolov5s",pretrained=True)
#joblib.dump(model_obj,"yolov5s.joblib")

def detect_obj(video_path):
    try:
        cap = cv2.VideoCapture(video_path)

        # Get Frame counts
        frame_count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # Get FPS
        FPS=cap.get(cv2.CAP_PROP_FPS)
        # skip window
        skip_window=int(FPS)

        # video len
        vid_len=frame_count/FPS
        thresh=int((vid_len//3)*3)
        # print(frame_count,thresh)

        # Open a text file for storing object names
    
        blist=[]
        d_obj=[]
        fcount=0
        while fcount<(thresh):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fcount * skip_window)
            fcount+=1
            print(fcount,"--------------------------")
            ret, frame = cap.read()
            if not ret:
                break

            height, width, channels = frame.shape

            results = model_obj(frame)  # inference
            outs=results.pandas().xyxy[0][["name","confidence","xmin","ymin","xmax","ymax"]].values.tolist()

            #
            #print(outs)
            for out in outs:
                x,y,w,h=out[2],out[3],1,1
                obj=out[0]
                conf=out[1]
                if conf>=CONFIDENCE:
                    blist.append(obj)
                else:
                    blist.append("")

            # for every 3 sec reinitialize the list
            if  fcount%3==0:
                d_obj.append(blist)
                blist=[]

        
    except Exception as e:
        print(e)
    finally:
        # Release video capture and writer, and close the text file
        cap.release()
    
        cv2.destroyAllWindows()
        # print(d_obj)
        return d_obj