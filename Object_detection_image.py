######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained neural network to perform object detection.
# It loads the classifier and uses it to perform object detection on an image.
# It draws boxes, scores, and labels around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
IMAGE_NAME = r'0_Howard, renewal.pdf.jpg'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to image
print(CWD_PATH,"-------------------------------------------------------")
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 6

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#print(detection_boxes)
# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
#print(detection_scores)
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
image = cv2.imread(PATH_TO_IMAGE)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_expanded = np.expand_dims(image_rgb, axis=0)
height,width = image.shape[:2]
print(width, height)
# Perform the actual detection by running the model with the image as input
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

# Draw the results of the detection (aka 'visulaize the results')

#print(scores,"****************SCROES***********************")

vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=0.60)
min_score_thresh=0.60
true_boxes = boxes[0][scores[0] > min_score_thresh]
corlst = []
for i in range(true_boxes.shape[0]):
    ymin = true_boxes[i,0]*height
    xmin = true_boxes[i,1]*width
    ymax = true_boxes[i,2]*height
    xmax = true_boxes[i,3]*width
    y0 = ymin
    y1 = ymax
    x0 = xmin
    x1 = xmax
    x0 = int(x0)
    y0 = int(y0)
    x1 = int(x1)
    y1 = int(y1)
    print(x0)
    print(type(y0))
    corlst.append([x0,y0,x1,y1])
    #print ("Top left")
    #print (xmin,ymin,)
    #print ("Bottom right")
    #print (xmax,ymax)
print(corlst)
# All the results have been drawn on image. Now display the image.
cv2.imwrite(r'0_OUTPUT\25sample.jpg', image)
#cv2.imshow('Object detector', image)

# Press any key to close the image
cv2.waitKey(0)

# Clean up
#cv2.destroyAllWindows()
image  = cv2.imread(r"C:\Users\MANOMAY\Desktop\Mahesh\TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10-master\0_Howard, renewal.pdf.jpg")
color = (0,0,255)
thickness = 2
for i in range(len(corlst)):
    start_point = (corlst[i][0],corlst[i][1])
    end_point = (corlst[i][2],corlst[i][3])
    image1 = cv2.rectangle(image, start_point, end_point, color, thickness)
cv2.imwrite(r"0_OUTPUT\imkol123.jpg",image1)
color = (255,255,255)
thickness = -1
for i in range(len(corlst)):
    start_point = (corlst[i][0],corlst[i][1])
    end_point = (corlst[i][2],corlst[i][3])
    image1 = cv2.rectangle(image, start_point, end_point, color, thickness)
cv2.imwrite(r"0_OUTPUT\remov.jpg",image1)

img = cv2.imread(r"C:\Users\MANOMAY\Desktop\Mahesh\TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10-master\0_Howard, renewal.pdf.jpg")
for i in range(len(corlst)):
    crop_img = img[corlst[i][1]:corlst[i][3], corlst[i][0]:corlst[i][2]]
    cv2.imwrite(r"0_OUTPUT\Cropped\\"+str(i)+"_cropped.jpg", crop_img)


#import numpy as np
#import cv2

#image = cv2.imread(r'C:\Users\MANOMAY\Desktop\outputs\001.jpg',cv2.IMREAD_UNCHANGED)

#position = (265, 835)
#cv2.putText(
 #    image, #numpy array on which text is written
  #   "30", #text
   #  position, #position at which writing has to start
   #  cv2.FONT_HERSHEY_SIMPLEX, #font family
   #  2, #font size
   #  (0, 0, 0, 255), #font color
   #  3) #font stroke
cv2.imwrite(r'outpu12345t.jpg', image)