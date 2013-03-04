#title           :HumanManipulator.py
#description     :This class is an implementation of "Robot specific layer". The class implements Robot interface
#                 and made it user friendly for human manipulation with clothing. Human in this case 
#                 substitute robotic device.
#author          :Jan Sindler
#conact          :sidnljan@fel.cvut.cz
#date            :20130508
#version         :1.0
#usage           :cannot be used alone
#notes           :
#python_version  :2.7.3  
#==============================================================================

import roslib; roslib.load_manifest('contour_model_folding')
from RobInt import RobInt
import rospy
import cv
import logging
import sys
import numpy as np

# HumanManipulator class represent robotic manipualtor simulated by human being.
      
class HumanManipulator(RobInt):  

    lastImageIndex = 0
      
    def liftUp(self, liftPoints):
        print "Grasp and lift up following points:"
        if(liftPoints == None):
            print "Some of the grasp points wasn't set."
            return
        for pt in liftPoints:
            print pt
            
        cv.NamedWindow("Fold visualisation")
        img = self.getImageOfObsObject(self.lastImageIndex)
        unw_img = self.__unwarp_image(img)
        for pt in liftPoints:
            cv.Circle(unw_img,pt,3,cv.CV_RGB(255,0,0),2)               
        cv.ShowImage("Fold visualisation",unw_img)
        cv.WaitKey()
        cv.DestroyWindow("Fold visualisation")
        
        #raw_input("Hit key to continue")
                    
    def place(self, targPoints):
        print "Place grasped objects to following points:"
        if(targPoints == None):
            print "Some of the target points wasn't set."
            return
        for pt in targPoints:
            print pt
        cv.NamedWindow("Fold visualisation")
        img = self.getImageOfObsObject(self.lastImageIndex)
        unw_img = self.__unwarp_image(img)
        for pt in targPoints:
            intPt = (int(pt[0]),int(pt[1]))
            cv.Circle(unw_img,intPt,3,cv.CV_RGB(0,0,255),2)               
        cv.ShowImage("Fold visualisation",unw_img)
        cv.WaitKey()
        cv.DestroyWindow("Fold visualisation")
        #raw_input("Hit key to continue")


    ##  Load an image grabed by a camera.
    #
    #   Images are now stored on a local HDD
    #   @param index The index of image to be loaded
    #   @return The image loaded from a file        
    def getImageOfObsObject(self, index):
        logging.debug("TAKE_PICTURE - Begin")
        takenImage = None
        self.lastImageIndex = index
        """ take a picture from Kinect
        logging.debug("TAKE_PICTURE - Picture is from Kinect.")
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

        #""" take a picture from file
        logging.debug("TAKE_PICTURE - Picture is from a file.")
        #path = "/media/Data/clothImages/towel/imT_%02d.png" % index
        path = "/media/Data/clothImages/tShirt/im_%02d.png" % index
        
        try:
           takenImage = cv.LoadImage(path,cv.CV_LOAD_IMAGE_COLOR)
        except:
           print "File not found or cannot be loaded. Path = " + path
           sys.exit()
        #"""

        """ Show image from Kinect
        cv.NamedWindow("Image from Kinect")
        cv.ShowImage("Image from Kinect",takenImage)
        cv.WaitKey()
        #cv.DestroyWindow("Image from Kinect")
        #"""

        logging.debug("TAKE_PICTURE - End")
        return takenImage
        
    ## Compute and return homography between side and top view
    #
    #  It takes the current view into one directly above the table. Correspondence 
    #    between points was made by a hand.
    #  @return 3x3 homography matrix
    def get_homography(self):
        # set up source points (model points)
        srcPoints = cv.fromarray(np.matrix([[63, 343],[537, 367],[550, 137],[78, 123]], dtype=float))
        # set up destination points (observed object points)
        #dstPoints = cv.fromarray(np.matrix([[120,285],[420,359],[455,186],[228,143]], dtype=float))
        dstPoints = cv.fromarray(np.matrix([[22, 383],[608, 385],[541, 187],[100, 196]], dtype=float))
        # compute homography
        H = cv.CreateMat(3,3,cv.CV_32FC1)
        cv.FindHomography(srcPoints,dstPoints,H) #def. setting is [method=0,ransacReprojThreshold=3.0,status=None]
        return H 
        
        
#private support methods
        
    ## Remove perspective distortion from image. 
    #
    #   In fact this function creates a top view from side view.
    #   @param image An input image with perspective distortion
    #   @return Return top view image.
    def __unwarp_image(self,image):
        H = self.get_homography()
        # do the image center correciton
        
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
