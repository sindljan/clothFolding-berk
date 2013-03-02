#!/usr/bin/env python

#An package that provides folding process.
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

def main():
    img = take_picture(4)
    #compute a homography
    H = get_homography()
    #unwarped the image. Turn the image into the top view.
    unw_img = unwrap_image(img,H)
    # initialization
    background = thresholding.GREEN_BG
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
    
## Remove perspective distortion from image. 
#
#   In fact this function creates a top view from side view.
#   @param image An input image with perspective distortion
#   @param transformation The transformation that converts side view to top view
#   @return Return top view image.
def unwrap_image(image, transformation):
    H = transformation
    # calculate transformation between image centers
    src_center = [cv.GetSize(image)[0]/2,cv.GetSize(image)[1]/2]
    z = 1./(H[2,0]*src_center[0]+H[2,1]*src_center[1]+H[2,2])
    dstX = (H[0,0]*src_center[0]+H[0,1]*src_center[1]+H[0,2])*z
    dstY = (H[1,0]*src_center[0]+H[1,1]*src_center[1]+H[1,2])*z
    
    # now when we know corespondence between centres we can update transformation
    H[0,2] += src_center[0] - dstX
    H[1,2] += src_center[1] - dstY
    
    # do the transformation
    unw_img = cv.CreateImage((640,480),cv.IPL_DEPTH_8U,3)
    cv.WarpPerspective(image,unw_img,H, cv.CV_INTER_LINEAR+cv.CV_WARP_FILL_OUTLIERS+cv.CV_WARP_INVERSE_MAP, (0,0,0,0)) # pixels that are out of the image are set to black
    return unw_img
    
## Compute and return homography between side and top view
#
#  It takes the current view into one directly above the table. Correspondence 
#    between points was made by a hand.
#  @return 3x3 homography matrix
def get_homography():
    # set up source points (model points)
    srcPoints = cv.fromarray(np.matrix([[63, 343],[537, 367],[550, 137],[78, 123]], dtype=float))
    # set up destination points (observed object points)
    #dstPoints = cv.fromarray(np.matrix([[120,285],[420,359],[455,186],[228,143]], dtype=float))
    dstPoints = cv.fromarray(np.matrix([[22, 383],[608, 385],[541, 187],[100, 196]], dtype=float))
    # compute homography
    H = cv.CreateMat(3,3,cv.CV_32FC1)
    cv.FindHomography(srcPoints,dstPoints,H) #def. setting is [method=0,ransacReprojThreshold=3.0,status=None]
    return H  

    
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
    
    """ take a picture from file
    #path = "/media/Data/clothImages/towel/imA%02d.png" % index
    path = "/media/Data/clothImages/tShirt/im_%02d.png" % index
    #path = "/media/Data/clothImages/tShirt/imF_%02d.png" % index
    try:
       takenImage = cv.LoadImage(path,cv.CV_LOAD_IMAGE_COLOR)
    except:
       print "File not found or cannot be loaded. Path = " + path
       sys.exit()
    #"""
    
    #visualise
    """ DEBUG
    cv.NamedWindow("Image from Kinect")
    cv.ShowImage("Image from Kinect",takenImage)
    cv.WaitKey()
    #cv.DestroyWindow("Image from Kinect")
    #"""
    
    return takenImage


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException: pass
