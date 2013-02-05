#!/usr/bin/env python

#An package that provides folding process.
import roslib
import sys
import math
import rospy
import cv
import os.path
import pickle
import numpy as np
roslib.load_manifest("conture_model_folding")
from clothing_models import Models
from shape_window.ShapeWindow import *
from shape_window import ShapeWindowUtils
from shape_window import Geometry2D
from visual_feedback_utils import Vector2D, thresholding, shape_fitting

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

    def getFoldedModel(self,foldLine):
        #do the fold line
        noise = 0
        for pt in foldLine:
            numWithNoise = (pt[0] + noise, pt[1])
            self.foldline_pts.append(numWithNoise)
            noise = noise + 1
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
    img = takePicture(imgStartIndex)
    #compute a homography
    H = getHomography()
    #unwarped the image. Turn the image into the top view.
    unw_img = cv.CloneImage(img)
    cv.WarpPerspective(img,unw_img,H, cv.CV_INTER_LINEAR+cv.CV_WARP_FILL_OUTLIERS+cv.CV_WARP_INVERSE_MAP, (255,255,255,255)) # pixels that are out of the image are set to white
    #Get initial model
    model = getInitialModel()
    #fit the model to the image
    model = fitModelToImage(model,unw_img)
    #for each desired fold
    NumOfFolds = 0
    if(TYPE == ASYMM):
        NumOfFolds = 2
    for i in range(1,NumOfFolds+1):
        showMessage("Do fold num: %02.d from %02.d" %(i,NumOfFolds), MsgTypes.info)
        #create a fold
        L = getFoldLine(model,i);
        #create a new model with fold
            #print str(model.polygon_vertices_int())
        foldedModel = createFoldedModel(model,unw_img,L)
            #print str(model.polygon_vertices_int())
        #excute a fold
        if(executeFold(model,foldedModel,L) != FoldResults.succesfull):
            return 1
        model = foldedModel
        #take an image
        img = takePicture(imgStartIndex + i)
        #unwarp image
        unw_img = cv.CloneImage(img)
        cv.WarpPerspective(img,unw_img,H, cv.CV_INTER_LINEAR+cv.CV_WARP_FILL_OUTLIERS+cv.CV_WARP_INVERSE_MAP, (255,255,255,255)) # pixels that are out of the image are set to white
        #fit the new model to the image
        #model = fitModelToImage(model,unw_img)

## Execute fold according to the fold line
#
#   @param model The model of current state of an obseved object
#   @param foldedModel The model how an observed object would look like after fold
#   @param foldLine The fold line.
#   @return Return how succesfull the folding process was.
def executeFold(model,foldedModel,foldLine):
    # selected points to be grasped
    #raw_input("before getGraspPoints...")
    gps = getGraspPoints(model,foldedModel,foldLine)
    if( gps == None):
        return FoldResults.noGraspedPoints
    # deffine a new positin of that points
    #raw_input("before getNewPositionOfGraspPoints...")
    np_gps = getNewGraspPointsPosition(gps,foldLine)
    # this part would be done by my hand. literally
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
def getNewGraspPointsPosition(points,foldLine):
    mirrored_pts = []
    for pt in points:
        mirrored_pts.append(Vector2D.mirror_pt(pt,foldLine))
        
    showMessage("Move grasped points to: " + str(mirrored_pts), MsgTypes.info)
    return mirrored_pts

## Return a list of points to be grasped by a robot
#
#   The points that are not on the same place in model and foldedModel 
#   are candidates to grasping points.
#   @param model The model of current state of an obseved object
#   @param foldedModel The model how an observed object would look like after fold
#   @return List of tuples that represents a position(int the image) of points to be grasped.
def getGraspPoints(model,foldedModel,foldLine): 
    gps = []
    pointsInModel = model.polygon_vertices_int()
    pointsInFoldedModel = foldedModel.polygon_vertices_int()
    showMessage("Model points: " + str(pointsInModel), MsgTypes.debug)
    showMessage("Folded model points: " + str(pointsInFoldedModel), MsgTypes.debug)
    
    # select a points that changes a place
    for pt in pointsInModel:
        try:
            pointsInFoldedModel.index(pt)
        except:
            gps.append(pt)
    showMessage("Grasped points: " + str(gps), MsgTypes.debug)
    
    if(len(gps) >= 2):
        # select two points with the biggest distance (eucledian) from fold line
        fmd = 0; # first max distance
        smd = 0; # second max distance
        fmdp = None;
        smdp = None;
        for pt in gps:
            #dist = Geometry2D.ptLineDistance(tupleToPoint(pt),tuplesToLine(foldLine))
            dist = getDist(pt,foldLine)
            showMessage( "Distant between point and line = " + str(dist) + " line = " + str(foldLine) + " point = " + str(pt), MsgTypes.debug)
            if(dist > fmd):
                fmd = dist
                fmdp = pt
            elif(dist > smd):
                smd = dist
                smdp = pt
        gps = [fmdp,smdp]
    elif(len(gps) < 2):
         gps = None
    
    showMessage("Selected grasped points: " + str(gps), MsgTypes.info)
    return gps

def getDist(pt,foldLine):
    #www.mathworld.wolfram.com/Point-LineDistance2-Dimensional.html
    pt1 = foldLine[0] 
    pt2 = foldLine[1] 
    dist = abs((pt2[0]-pt1[0])*(pt1[1]-pt[1])-(pt1[0]-pt[0])*(pt2[1]-pt1[1]))/math.sqrt(pow((pt2[0]-pt1[0]),2)+pow((pt2[1]-pt1[1]),2))
    return dist

def tuplesToLine(pts):
    pt1 = pts[0]
    pt2 = pts[1]
    return Geometry2D.Line(tupleToPoint(pt1),tupleToPoint(pt2))
    
def tupleToPoint(pt):
    return Geometry2D.Point(pt[0],pt[1])
   
## Creates fold according to the model current state and order of fold
#
#   Hardcoded fold definition for each of known objects
#   @param model The model fitted to observed object.
#   @param i Index that defines fold order
#   @return fold line as two points in 2D space
def getFoldLine(model,i):
    foldStart = None
    foldEnd = None
    
    if(TYPE == ASYMM):
        #towel
        if(i == 1):
            #Fold in half
            showMessage("Model verticies " + str(model.polygon_vertices_int()), MsgTypes.info)
            [bl,tl,tr,br] = [Geometry2D.Point(int(pt[0]), int(pt[1])) for pt in model.polygon_vertices_int()]
            # shift in x direction otherwise a model with fold would be illegal
            bl.translate(0,-1)
            br.translate(0,-1)
            tl.translate(0,1)
            tr.translate(0,1)
            
            foldStart = Geometry2D.LineSegment(bl,br).center().toTuple() #NOT OPTIMAL
            foldEnd = Geometry2D.LineSegment(tl,tr).center().toTuple() #NOT OPTIMAL
        elif(i == 2):
            #Fold in half again
            showMessage("Model verticies " + str(model.polygon_vertices_int()), MsgTypes.info);
            [tr,tl,bl,br,a,a,a,a] = [Geometry2D.Point(int(pt[0]), int(pt[1])) for pt in model.polygon_vertices_int()]
            bl.translate(-3,0)
            br.translate(3,0)
            tl.translate(-3,0)
            tr.translate(3,0)
            
            foldStart = Geometry2D.LineSegment(br,tr).center().toTuple() #NOT OPTIMAL
            foldEnd = Geometry2D.LineSegment(bl,tl).center().toTuple() #NOT OPTIMAL
    else:
        showMessage("Not implemented type of cloth",MsgTypes.exception)
        sys.exit()
        
    foldLine = [foldStart, foldEnd]
    showMessage("New fold line: " + str(foldLine),MsgTypes.info)
    return foldLine;

##  Create a new model by folding the old one.
#
#   The function createFoldedModel takes an unfolded model on its input 
#   and image of the folded object. The function visualise both inputs 
#   in a graphical window. User would create a new foldline here.
#   @param model A model to be folded
#   @param image An image of folded object
#   @return A new model with fold.
def createFoldedModel(_model, _image, _foldLine):

    fm = FoldMaker(_model,_image)
    modelWithFold = fm.getFoldedModel(_foldLine)
    
    if(modelWithFold == None):
        sys.exit()
        
    #""" 
    print "/**************Test****************/"
    cv.NamedWindow("Debug window")
    img = cv.CloneImage(_image)
    cv.PolyLine(img,[modelWithFold.polygon_vertices_int()],1,cv.CV_RGB(255,0,0),1)               
    cv.ShowImage("Debug window",img)
    cv.WaitKey()
    cv.DestroyWindow("Debug window")
    print "/************EndOfTest*************/"
    #"""
    
    showMessage("Verticies of a model with fold: " + str(modelWithFold.polygon_vertices_int()), MsgTypes.info);
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
def fitModelToImage(model,image):
    showMessage("Model fitter has started.", MsgTypes.info);
    # initialization
    background = thresholding.WHITE_BG
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
    showMessage("Model verticies after fitting: " + str(model.polygon_vertices_int()), MsgTypes.info);
    showMessage("Model fitter finished.", MsgTypes.info);
    
    return fitted_model
        
##  Load an initial model from HDD
#
#   @return Model of observed object
def getInitialModel():
    #unpicle model from file
    modelPath = "/media/Data/models/im_towel.pickle"
    initialModel = pickle.load(open(modelPath))
    return initialModel
    
##  Load an image grabed by a camera.
#
#   Images are now stored on a local HDD
#   @param index The index of image to be loaded
#   @return The image loaded from a file
def takePicture(index):
    path = "/media/Data/clothImages/towel/im%02d.JPG" % index
    try:
        img = cv.LoadImage(path,cv.CV_LOAD_IMAGE_COLOR)
    except:
        showMessage("File not found or cannot be loaded. Path = " + path, MsgTypes.exception)
        sys.exit()
    showMessage("Loading image from the file " + path, MsgTypes.info)
    return img

## Compute and return homography between side and top view
#
#  It takes the current view into one directly above the table. Correspondence 
#    between points was made by a hand.
#  @return 3x3 homography matrix
def getHomography():
    # set up source points
    srcPoints = cv.fromarray(np.matrix([[203,374],[432,376],[431,137],[201,139]], dtype=float))
    # set up destination points
    dstPoints = cv.fromarray(np.matrix([[120,285],[420,359],[455,186],[228,143]], dtype=float))
    # compute homography
    H = cv.CreateMat(3,3,cv.CV_32FC1)
    cv.FindHomography(srcPoints,dstPoints,H) #def. setting is [method=0,ransacReprojThreshold=3.0,status=None]
    showMessage("Computed homography" + str(np.asarray(H)), MsgTypes.info)
    return H
    
## Show a message according to the setup verbosity and append a proper label
#
#  @param text Text of the message
#  @param msgType Type of message. One of the elements of MsgTypes class.
def showMessage(text,msgType):
    if(msgType == MsgTypes.info):
        print "INFO: " + text
    elif(msgType == MsgTypes.debug):
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
    return parser.parse_args()
    

if __name__ == '__main__':
    args = parse()
    try:
        main(args)
    except rospy.ROSInterruptException: pass
