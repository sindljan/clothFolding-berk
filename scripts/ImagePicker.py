#!/usr/bin/env python

##    @package conture_model_folding
# Stores images from Kinect to HDD to defined folder

import roslib; roslib.load_manifest('contour_model_folding')
import sys
import math
import rospy
import cv
import os.path
import pickle
import numpy as np
import tf
from clothing_models import Models
from shape_window.ShapeWindow import *
from shape_window import ShapeWindowUtils
from shape_window import Geometry2D
from visual_feedback_utils import Vector2D, thresholding, shape_fitting
from sensor_msgs.msg import Image
from conture_model_folding.srv import *
from cv_bridge import CvBridge, CvBridgeError

def main(args):
    # a mask for image names look like:
    #   folder/im%suffix_%start_index.png
    starting_index = int(args[0]) 
    suffix = args[1] 
    folder = args[2]
    
    cv.NamedWindow("Image from Kinect")
    cv.WaitKey()
    
    img = take_picture(1)
    # show it 
    #visualise
    #"""
    
    cv.ShowImage("Image from Kinect",img)
    cv.WaitKey()
    cv.DestroyWindow("Image from Kinect")
    #"""
    
    # store it
    cv.SaveImage("%s/im%s_%02d.png"%(folder,suffix,starting_index),img)

    
##  Load an image grabed by a camera.
#
#   Images are now stored on a local HDD
#   @param index The index of image to be loaded
#   @return The image loaded from a file
def take_picture(index):
    print "TAKE_PICTURE"
    takenImage = None
    
    #"""
    #take a picture from Kinect
    rospy.wait_for_service('get_kinect_image')
    try:
        service = rospy.ServiceProxy('get_kinect_image',GetImage) #create service
        msg_resp = service() #call service and return image
        imData = msg_resp.image
    except rospy.ServiceException, e:
        show_message("Image grabing service failed: %s."%e, MsgTypes.exception)
        return None
        
    #convert it to format accepted by openCV
    try:
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv(imData,"bgr8")
    except CvBridgeError, e:
        show_message("Image conversion error: %s."%e, MsgTypes.exception)
        return None
    
    #crop image
    roi = (0,0,620,480) # x,y(from the top of the image),width,height
    cropped = cv.GetSubRect(image,roi)
    takenImage = cv.CreateImage(roi[2:],cv.IPL_DEPTH_8U,3);
    cv.Copy(cropped,takenImage)
    #cv.SaveImage("./im.png",takenImage)
    #"""

    return takenImage

if __name__ == '__main__':
    args = sys.argv[1:]
    try:
        main(args)
    except rospy.ROSInterruptException: pass
