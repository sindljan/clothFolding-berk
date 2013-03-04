#!/usr/bin/env python
#title           :ImageReaderService.py
#description     :ROS service that subscribe images from Kinect device and keep last one in memory.
#                 Somebody then can ask for the last image.
#author          :Jan Sindler
#conact          :sidnljan@fel.cvut.cz
#date            :20130508
#version         :1.0
#usage           :./ImageReaderService.py
#notes           :
#python_version  :2.7.3  
#==============================================================================

import roslib; roslib.load_manifest('contour_model_folding')
import rospy
from sensor_msgs.msg import Image
from conture_model_folding.srv import *

grabedImage = None

def store_image(image):
    global grabedImage 
    grabedImage = image

def handle_get_image_request(some):
    print str(some)
    return GetImageResponse(grabedImage);

def main():
    rospy.init_node('image_reader_service')
    s = rospy.Service('get_kinect_image', GetImage, handle_get_image_request)
    rospy.Subscriber("/camera/rgb/image_color", Image, store_image)
    rospy.spin()
    
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException: pass
