#!/usr/bin/env python

#An package that provides folding process.
import roslib; roslib.load_manifest('conture_model_folding')
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

## Compute and return homography between side and top view
#
#  It takes the current view into one directly above the table. Correspondence 
#    between points was made by a hand.
#  @return 3x3 homography matrix
def get_homography():
    # set up source points (model points)
    srcPoints = cv.fromarray(np.matrix([[203,374],[432,376],[431,137],[201,139]], dtype=float))
    # set up destination points (observed object points)
    #dstPoints = cv.fromarray(np.matrix([[120,285],[420,359],[455,186],[228,143]], dtype=float))
    dstPoints = cv.fromarray(np.matrix([
        [173, 303],
        [289, 302],
        [284, 239],
        [184, 235]
        ], dtype=float))
    # compute homography
    H = cv.CreateMat(3,3,cv.CV_32FC1)
    cv.FindHomography(srcPoints,dstPoints,H) #def. setting is [method=0,ransacReprojThreshold=3.0,status=None]
    return H
## End of Support classes ----------------------------------------------    

def main():
    img = take_picture(1)
    #compute a homography
    H = get_homography()
    #unwarped the image. Turn the image into the top view.
    unw_img = cv.CreateImage((800,600),cv.IPL_DEPTH_8U,3)
    cv.WarpPerspective(img,unw_img,H, cv.CV_INTER_LINEAR+cv.CV_WARP_FILL_OUTLIERS+cv.CV_WARP_INVERSE_MAP, (255,255,255,255)) # pixels that are out of the image are set to white
    # initialization
    background = thresholding.WHITE_BG
    #Use the thresholding module to get the contour out
    shape_contour = thresholding.get_contour(unw_img,bg_mode=background,filter_pr2=False,crop_rect=None)
    #"""
    cv.NamedWindow("Debug window")
    img = cv.CloneImage(unw_img)
    cv.PolyLine(img,[shape_contour],1,cv.CV_RGB(0,0,255),1)               
    cv.ShowImage("Debug window",img)
    cv.WaitKey()
    cv.DestroyWindow("Debug window")
    #"""
    
##  Load an image grabed by a camera.
#
#   Images are now stored on a local HDD
#   @param index The index of image to be loaded
#   @return The image loaded from a file
def take_picture(index):
    print "TAKE_PICTURE"
    takenImage = None
    
    """
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
    roi = (80,80,480,400)
    cropped = cv.GetSubRect(image,roi)
    takenImage = cv.CreateImage(roi[2:],cv.IPL_DEPTH_8U,3);
    cv.Copy(cropped,takenImage)
    #cv.SaveImage("./im.png",takenImage)
    #"""
    
    #""" take a picture from file
    path = "/media/Data/clothImages/towel/imA%02d.png" % index
    try:
       takenImage = cv.LoadImage(path,cv.CV_LOAD_IMAGE_COLOR)
    except:
       print "File not found or cannot be loaded. Path = " + path
       sys.exit()
    #"""
    
    #visualise
    #""" DEBUG
    print "/**************Test****************/"
    cv.NamedWindow("Image from Kinect")
    cv.ShowImage("Image from Kinect",takenImage)
    cv.WaitKey()
    cv.DestroyWindow("Image from Kinect")
    print "/************EndOfTest*************/"
    #"""
    
    return takenImage


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException: pass
