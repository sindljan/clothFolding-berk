import roslib; roslib.load_manifest('contour_model_folding')
from RobInt import RobInt
import rospy
import cv
import logging

# HumanManipulator class represent robotic manipualtor simulated by human being.
      
class HumanManipulator(RobInt):  
      
    def liftUp(self, liftPoints):
        print "Grasp and lift up following points:"
        if(liftPoints == None):
            print "Some of the grasp points wasn't set."
            return
        for pt in liftPoints:
            print pt
        raw_input("Hit key to continue")
            
    def place(self, targPoints):
        print "Place grasped objects to following points:"
        if(targPoints == None):
            print "Some of the target points wasn't set."
            return
        for pt in targPoints:
            print pt
        raw_input("Hit key to continue")


    ##  Load an image grabed by a camera.
    #
    #   Images are now stored on a local HDD
    #   @param index The index of image to be loaded
    #   @return The image loaded from a file        
    def getImageOfObsObject(self, index):
        logging.debug("TAKE_PICTURE - Begin")
        takenImage = None

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
        path = "/media/Data/clothImages/towel/imT_%02d.png" % index
        #path = "/media/Data/clothImages/tShirt/im_%02d.png" % index
        
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
