#!/usr/bin/env python

##    @package cloth_models 
# 	  Define shape models by hand.
import math
import roslib
import sys
roslib.load_manifest("clothing_models")
import rospy
from numpy import *
import math
import cv
import os.path
from shape_window.ShapeWindow import *
from shape_window import Geometry2D

class HomoEstim(ShapeWindow):
    
    def __init__(self,filepath):
        bgd = cv.LoadImage(filepath)
        size = (bgd.width,bgd.height)
        ShapeWindow.__init__(self,name="Window",size=size)
        self.shapesLock.acquire()
        self.img = cv.CloneImage(bgd)
        self.background = cv.CloneImage(bgd)
        self.shapesLock.release()
    
    def initExtended(self):
        closeButton = CVButton(text="CLOSE",bottomLeft=Geometry2D.Point(350,100), onClick = self.close)
        self.addOverlay(closeButton)
        
    def onMouse(self,event,x,y,flags,param):
        if event==cv.CV_EVENT_LBUTTONUP:
            print str([x,y])
    
def main(args):

    imagepath = args[0]

    w = HomoEstim(imagepath)
    while(not w.isClosed()):
        rospy.sleep(0.05)
    return
    
if __name__ == '__main__':
    args = sys.argv[1:]
    try:
        main(args)
    except rospy.ROSInterruptException: pass
