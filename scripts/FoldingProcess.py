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

ASYMM = 0 			# Asymmetric polygon model
#SYMM = 1 			# Symmetric polygon model
#SWEATER_SKEL = 2 	# Sweater model
#TEE_SKEL = 3 		# Tee model
#PANTS_SKEL = 4 		# Pants model
#SOCK_SKEL = 5 		# Sock model

TYPE = ASYMM 	#Adjust to change which type of model is being created

## Begin of support classes --------------------------------------------

class FoldResults:
    succesfull = 0
    noGraspedPoints = 1
    
class MsgTypes:
    info = 1
    debug = 2
    exception = 3

class FoldMaker:   
        
    def __init__(self,_model,_image):
        bgd = cv.CloneImage(_image)
        self.background = bgd
        self.initial_model = _model
        self.initial_model.set_image(bgd)
        self.foldline_pts = []
        self.foldline = None
        cv.NamedWindow("Fold visualisation")
        cv.PolyLine(self.background,[_model.polygon_vertices_int()],1,cv.CV_RGB(255,0,0),2)
        cv.ShowImage("Fold visualisation",self.background )

    def get_folded_model(self,foldLine):
        #do the fold line
        # A function Vector2D.intercept doesnt work if x or y of pts are same. Therofre I put some noise if neded
        noise = -1;
        difX = (foldLine[0])[0] - (foldLine[1])[0]
        difY = (foldLine[0])[1] - (foldLine[1])[1]
        if ((difX == 0) and (difY == 0)):
            self.foldline_pts.append( ((foldLine[0])[0]+noise,(foldLine[0])[1]+noise) )
        elif(difX == 0):
            self.foldline_pts.append( ((foldLine[0])[0]+noise,(foldLine[0])[1]) )
        elif(difY == 0):
            self.foldline_pts.append( ((foldLine[0])[0],(foldLine[0])[1]+noise) )
        else:
            self.foldline_pts.append(foldLine[0])
        self.foldline_pts.append(foldLine[1])
        
        
        self.foldline = Vector2D.make_ln_from_pts(self.foldline_pts[0],self.foldline_pts[1])
        ln_start = Vector2D.intercept(self.foldline,Vector2D.horiz_ln(y=0))
        ln_end = Vector2D.intercept(self.foldline,Vector2D.horiz_ln(y=self.background.height))

        #visualisation
        cv.Line(self.background,(int(ln_start[0]),int(ln_start[1])),(int(ln_end[0]),int(ln_end[1])),cv.CV_RGB(0,0,0))
        cv.Circle(self.background,self.foldline_pts[0],4,cv.CV_RGB(0,255,0))
        cv.Circle(self.background,self.foldline_pts[1],4,cv.CV_RGB(0,255,0))
        cv.Circle(self.background,(int(ln_start[0]),int(ln_start[1])),4,cv.CV_RGB(255,0,0))
        cv.Circle(self.background,(int(ln_end[0]),int(ln_end[1])),4,cv.CV_RGB(255,0,0))
        cv.ShowImage("Fold visualisation",self.background )
        cv.WaitKey()
        cv.DestroyWindow("Fold visualisation")
        
        model = Models.Point_Model_Folded(self.initial_model,self.foldline_pts[0],self.foldline_pts[1])
        model.draw_to_image(self.background,cv.RGB(255,0,0))
        if model.illegal() or model.structural_penalty() >= 1.0:
            print "Model is illegal!"
            return None
        else:
            return model

## End of Support classes ----------------------------------------------    

def main(args):
    imgStartIndex = 1
    #take an initial image from camera
    img = take_picture(imgStartIndex)
    #compute a homography
    H = get_homography()
    #unwarped the image. Turn the image into the top view.
    unw_img = unwrap_image(img,H)
    #Get initial model
    model = get_initial_model()
    #fit the model to the image
    model = fit_model_to_image(model,unw_img)
    #for each desired fold
    NumOfFolds = 0
    if(TYPE == ASYMM):
        NumOfFolds = 2
    for i in range(1,NumOfFolds+1):
        show_message("Do fold num: %02.d from %02.d" %(i,NumOfFolds), MsgTypes.info)
        #create a fold
        L = get_fold_line(model,i);
        #create a new model with fold
        foldedModel = create_folded_model(model,unw_img,L)
        #excute a fold
        if(execute_fold(model,foldedModel,L) != FoldResults.succesfull):
            return 1
        model = foldedModel
        #take an image
        img = take_picture(imgStartIndex + i)
        #unwarp image
        unw_img = cv.CloneImage(img)
        cv.WarpPerspective(img,unw_img,H, cv.CV_INTER_LINEAR+cv.CV_WARP_FILL_OUTLIERS+cv.CV_WARP_INVERSE_MAP, (255,255,255,255)) # pixels that are out of the image are set to white
        #fit the new model to the image
        model = fit_model_to_image(model,unw_img)

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
   
## Execute fold according to the fold line
#
#   @param model The model of current state of an obseved object
#   @param foldedModel The model how an observed object would look like after fold
#   @param foldLine The fold line.
#   @return Return how succesfull the folding process was.
def execute_fold(model,foldedModel,foldLine):
    # selected points to be grasped
    #raw_input("before get_grasp_points...")
    gps = get_grasp_points(model,foldedModel,foldLine)
    if( gps == None):
        return FoldResults.noGraspedPoints
    # deffine a new positin of that points
    #raw_input("before getNewPositionOfGraspPoints...")
    np_gps = get_new_grasp_points_position(gps,foldLine)
    # this part would be done by my hand. literally
    raw_input("Do the fold and hit enter...")
        # grasped the points
        # move the grasped points to the defined position
        # ungrasp it
    return FoldResults.succesfull

## Return a list of new position of greasped points.
#
#   This funciton mirror the points accordigng to the fold line.
#   @param points The point to be translate.
#   @param foldLine The fold line for mirroring.
#   @return List of tuples that represents a new position(int the image) of grasped points. 
def get_new_grasp_points_position(points,foldLine):
    mirrored_pts = []
    for pt in points:
        if(pt != None):
            mirrored_pts.append(Vector2D.mirror_pt(pt,foldLine))
        else:
            show_message("Some of the grasp points wasn't set.", MsgTypes.exception)
        
    show_message("Move grasped points to: " + str(mirrored_pts), MsgTypes.info)
    return mirrored_pts

## Return a list of points to be grasped by a robot
#
#   The points that are not on the same place in model and foldedModel 
#   are candidates to grasping points.
#   @param model The model of current state of an obseved object
#   @param foldedModel The model how an observed object would look like after fold
#   @return List of tuples that represents a position(int the image) of points to be grasped.
def get_grasp_points(model,foldedModel,foldLine): 
    gps = []
    pointsInModel = model.polygon_vertices_int()
    pointsInFoldedModel = foldedModel.polygon_vertices_int()
    show_message("Model points: " + str(pointsInModel), MsgTypes.debug)
    show_message("Folded model points: " + str(pointsInFoldedModel), MsgTypes.debug)
    
    # select a points that changes a place
    for pt in pointsInModel:
        try:
            pointsInFoldedModel.index(pt)
        except:
            gps.append(pt)
    show_message("Candidates to grasped points: " + str(gps), MsgTypes.debug)
    
    if(len(gps) >= 2):
        # select two points with the biggest distance (eucledian) from fold line
        fmd = 0; # first max distance
        smd = 0; # second max distance
        fmdp = None;
        smdp = None;
        for pt in gps:
            #dist = Geometry2D.ptLineDistance(tuple_to_point(pt),tuples_to_line(foldLine))
            dist = point_line_distance(pt,foldLine)
            show_message( "Distant between point and line = " + str(dist) + " line = " + str(foldLine) + " point = " + str(pt), MsgTypes.debug)
            if(dist > fmd):
                fmd = dist
                fmdp = pt
            elif(dist > smd):
                smd = dist
                smdp = pt
        gps = [fmdp,smdp]
    elif(len(gps) < 2):
         gps = None
    
    show_message("Selected grasped points: " + str(gps), MsgTypes.info)
    return gps

def point_line_distance(pt,foldLine):
    #www.mathworld.wolfram.com/Point-LineDistance2-Dimensional.html
    pt1 = foldLine[0] 
    pt2 = foldLine[1] 
    dist = abs((pt2[0]-pt1[0])*(pt1[1]-pt[1])-(pt1[0]-pt[0])*(pt2[1]-pt1[1]))/math.sqrt(pow((pt2[0]-pt1[0]),2)+pow((pt2[1]-pt1[1]),2))
    return dist

def tuples_to_line(pts):
    pt1 = pts[0]
    pt2 = pts[1]
    return Geometry2D.Line(tuple_to_point(pt1),tuple_to_point(pt2))
    
def tuple_to_point(pt):
    return Geometry2D.Point(pt[0],pt[1])
   
## Creates fold according to the model current state and order of fold
#
#   Hardcoded fold definition for each of known objects
#   @param model The model fitted to observed object.
#   @param i Index that defines fold order
#   @return fold line as two points in 2D space
def get_fold_line(model,i):
    foldStart = None
    foldEnd = None
    
    if(TYPE == ASYMM):
        #towel
        if(i == 1):
            #Fold in half
            show_message("Model verticies " + str(model.polygon_vertices_int()), MsgTypes.info)
            [bl,tl,tr,br] = [Geometry2D.Point(int(pt[0]), int(pt[1])) for pt in model.polygon_vertices_int()]
            # shift in x direction otherwise a model with fold would be illegal
            bl.translate(0,-10)
            br.translate(0,-10)
            tl.translate(0,10)
            tr.translate(0,10)
            
            foldStart = Geometry2D.LineSegment(bl,br).center().toTuple() #NOT OPTIMAL
            foldEnd = Geometry2D.LineSegment(tl,tr).center().toTuple() #NOT OPTIMAL
        elif(i == 2):
            #Fold in half again
            show_message("Model verticies " + str(model.polygon_vertices_int()), MsgTypes.info);
            [tr,tl,bl,br] = ([Geometry2D.Point(int(pt[0]), int(pt[1])) for pt in model.polygon_vertices_int()])[0:4]
            bl.translate(-10,0)
            br.translate(10,0)
            tl.translate(-10,0)
            tr.translate(10,0)
            
            foldStart = Geometry2D.LineSegment(br,tr).center().toTuple() #NOT OPTIMAL
            foldEnd = Geometry2D.LineSegment(bl,tl).center().toTuple() #NOT OPTIMAL
    else:
        show_message("Not implemented type of cloth",MsgTypes.exception)
        sys.exit()
        
    foldLine = [foldStart, foldEnd]
    show_message("New fold line: " + str(foldLine),MsgTypes.info)
    return foldLine;

##  Create a new model by folding the old one.
#
#   The function create_folded_model takes an unfolded model on its input 
#   and image of the folded object. The function visualise both inputs 
#   in a graphical window. User would create a new foldline here.
#   @param model A model to be folded
#   @param image An image of folded object
#   @return A new model with fold.
def create_folded_model(_model, _image, _foldLine):
    show_message("CREATE FOLDED MODEL", MsgTypes.debug)
    fm = FoldMaker(_model,_image)
    modelWithFold = fm.get_folded_model(_foldLine)
    
    if(modelWithFold == None):
        sys.exit()
        
    """ 
    print "/**************Test****************/"
    cv.NamedWindow("Debug window")
    img = cv.CloneImage(_image)
    cv.PolyLine(img,[modelWithFold.polygon_vertices_int()],1,cv.CV_RGB(255,0,0),1)               
    cv.ShowImage("Debug window",img)
    cv.WaitKey()
    cv.DestroyWindow("Debug window")
    print "/************EndOfTest*************/"
    #"""
    
    show_message("Verticies of a model with fold: " + str(modelWithFold.polygon_vertices_int()), MsgTypes.info);
    return modelWithFold
    
##  Fit a model to an image
#
#   This function fits an input model to an input image. Using algorithm 
#   developed at Berkley. The function computes an object shape from the 
#   input image. The model is fitted to the object shape. The function computes
#   energy function from difference between model and object shape and try
#   to minimaze it.
#   @param model The model that has to be fitted.
#   @param image the image that has to be fitted.
#   @return Fitted model
def fit_model_to_image(model,image):
    show_message("FIT MODEL TO IMAGE", MsgTypes.debug)
    # initialization
    background = thresholding.GREEN_BG
    silent = True # true = silent, false = verbose
    show_graphics = True
    num_iters = 50
    
    #Properly set phases
    orient_opt      = True
    symm_opt        = True
    asymm_opt       = True
    fine_tuning_opt = True
    
    #Create an image to output
    image_out = cv.CloneImage(image)
    #Use the thresholding module to get the contour out
    shape_contour = thresholding.get_contour(image,bg_mode=background,filter_pr2=False,crop_rect=None)
    #"""
    cv.NamedWindow("Debug window")
    img = cv.CloneImage(image)
    cv.PolyLine(img,[shape_contour],1,cv.CV_RGB(0,0,255),1)               
    cv.ShowImage("Debug window",img)
    cv.WaitKey()
    cv.DestroyWindow("Debug window")
    #"""
    
    #Use the shaper fitter module to fit the model to image
    fitter = shape_fitting.ShapeFitter(     ORIENT_OPT=orient_opt,  SYMM_OPT=symm_opt,   
                                            ASYMM_OPT=asymm_opt,    FINE_TUNE=fine_tuning_opt,
                                            SILENT=silent,          SHOW=show_graphics,
                                            num_iters=num_iters )
    (nearest_pts, final_model, fitted_model) = fitter.fit(model,shape_contour,image_out,image)   
    """
    print "/**************Test****************/"
    cv.NamedWindow("Debug window")
    cv.PolyLine(image,[fitted_model.polygon_vertices_int()],1,cv.CV_RGB(0,255,0),1)               
    cv.ShowImage("Debug window",image)
    cv.WaitKey()
    cv.DestroyWindow("Debug window")
    print "/************EndOfTest*************/"
    #"""
    fitted_model.set_image(None)
    show_message("Model verticies after fitting: " + str(fitted_model.polygon_vertices_int()), MsgTypes.info);
    
    return fitted_model
        
##  Load an initial model from HDD
#
#   @return Model of observed object
def get_initial_model():
    #unpicle model from file
    modelPath = "/media/Data/models/im_towel.pickle"
    initialModel = pickle.load(open(modelPath))
    return initialModel
    
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
    roi = (0,0,620,400) # x,y(from the top of the image),width,height
    cropped = cv.GetSubRect(image,roi)
    takenImage = cv.CreateImage(roi[2:],cv.IPL_DEPTH_8U,3);
    cv.Copy(cropped,takenImage)
    #cv.SaveImage("./im.png",takenImage)
    #"""
    
    """ take a picture from file
    path = "/media/Data/clothImages/towel/imA%02d.png" % index
    try:
       takenImage = cv.LoadImage(path,cv.CV_LOAD_IMAGE_COLOR)
    except:
       print "File not found or cannot be loaded. Path = " + path
       sys.exit()
    #"""
    
    #visualise
    #""" DEBUG
    cv.NamedWindow("Image from Kinect")
    cv.ShowImage("Image from Kinect",takenImage)
    cv.WaitKey()
    #cv.DestroyWindow("Image from Kinect")
    #"""
    
    return takenImage

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
    
## Show a message according to the setup verbosity and append a proper label
#
#  @param text Text of the message
#  @param msgType Type of message. One of the elements of MsgTypes class.
def show_message(text,msgType):
    if(msgType == MsgTypes.info):
        print "INFO: " + text
    elif(msgType == MsgTypes.debug):
        #return
        print "DEBUG: " + text
    elif(msgType == MsgTypes.exception):
        print "ERROR: " + text
    else:
        print "Uknown type: " + text
    
## Parse input arguments using argparse package.
#
#  @return A list of named arguments and its values
def parse():
    import argparse
    parser = argparse.ArgumentParser(description='Run folding process')                            
    #return parser.parse_args()
    return None
    

if __name__ == '__main__':
    args = parse()
    try:
        main(args)
    except rospy.ROSInterruptException: pass
