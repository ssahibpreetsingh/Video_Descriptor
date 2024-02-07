import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names.txt", "r") as f:                          # Load the class file coco.names.txt
    classes = [line.strip() for line in f.readlines()]          # Convert all classes in the list

layer_names = net.getUnconnectedOutLayersNames()                # Get index of the output layers. (82,94,106)

# Read the video file
video_path = "cars1.mp4"                                        # video file path
cap = cv2.VideoCapture(video_path)                              # VideoCapture object: for reading frames from video

# Define the codec and create a VideoWriter object to save the output
fourcc = cv2.VideoWriter_fourcc(*"XVID")                        # set the "XVID" codec (commonly used for writing MPEG files)
output_file = "output.avi"                                      # Output file of processed video
video_writer = cv2.VideoWriter(output_file, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Open a text file for storing object names
text_file = open("detected_objects.txt", "w")                   # Open/Create

while True:
    ret, frame = cap.read()                                     # ret is a boolean indicating whether the frame was successfully read, and frame is the frame itself
    if not ret:
        break

    height, width, channels = frame.shape

    # Convert the frame to a blob
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # 0.00392: Scale factor.
    # (416, 416): Size to which the input frame is resized.
    # (0, 0, 0): Mean values that are subtracted from the frame. These values are usually the mean RGB values of the dataset the model was trained on.
    # Color channels of frame should be swapped from BGR to RGB
    # crop=False: Frame should not be cropped.

    # Set the input to the network
    net.setInput(blob)

    # Get the output layer names and run forward pass
    outs = net.forward(layer_names)                             # giving the bounding boxes and associated probabilities.

    # Post-process the results
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:                                            # process each output from the YOLO network.
        for detection in out:                                   # detection corresponds to an object detected by the network
            scores = detection[5:]
            '''The first 4 elements of the array represent the center coordinates (x, y) 
               and the width and height of the bounding box. 
               The remaining elements (from index 5 onwards) are the class scores for each class'''
            
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:                                # Only detections with a confidence higher than 0.5 considered valid
                
                # calculate the center of the bounding box for the detected object
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                
                # calculate the width and height of the bounding box.
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate the top-left corner of the bounding box.
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)                      # Append the class ID
                confidences.append(float(confidence))           # Append confidence
                boxes.append([x, y, w, h])                      # append bounding box

    # Apply non-maximum suppression to eliminate redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # 0.5: This is the score_threshold parameter. Any bounding box with a confidence score less than this threshold will be discarded.
    ''' 0.4: This is the nms_threshold parameter. This is the Intersection over Union (IoU) threshold used for NMS. 
             Two bounding boxes are considered to be "overlapping" if their IoU is greater than this threshold. 
             During NMS, if two boxes are found to be overlapping, 
             the one with the lower confidence score is discarded.'''

    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            text_file.write(label + "\n")
    
            # Draw bounding box and label on the frame
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Write the frame to the output video
    video_writer.write(frame)

# Release video capture and writer, and close the text file
cap.release()
video_writer.release()
text_file.close()

cv2.destroyAllWindows()
