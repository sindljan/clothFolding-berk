import math
import random
from numpy import *

def pt_x(pt):
    return pt[0]

def pt_y(pt):
    return pt[1]
    
def make_pt(x,y):
    return (float(x),float(y))

def pt_sum(pt1,pt2):
    return (pt1[0]+pt2[0],pt1[1]+pt2[1])
    
def pt_diff(pt1,pt2):
    return (pt1[0]-pt2[0],pt1[1]-pt2[1])
    
def pt_distance(pt1,pt2):
    return vect_length(pt_diff(pt1,pt2))
    
def pt_scale(pt,scale):
    return (pt[0]*scale,pt[1]*scale)

def pt_center(pt1,pt2):
    return pt_scale(pt_sum(pt1,pt2),0.5)

def pt_seg_distance(pt,seg):
    #mathworld.wolfram.com/Point-LineDistance2-Dimensional.html
    (px,py) = pt
    (sx1,sy1) = seg[2]
    (sx2,sy2) = seg[3]
    dist = abs((sx2-sx1)*(sy1-py)-(sx1-px)*(sy2-sy1))/sqrt((sx2-sx1)**2+(sy2-sy1)**2)
    return dist
    
def dot_prod(pt1,pt2):
    return pt1[0]*pt2[0]+pt1[1]*pt2[1]
    
def vect_length(vect):
    return sqrt(dot_prod(vect,vect))
    
def make_ln_from_pts(pt1,pt2):
    vect = pt_diff(pt2,pt1)
    #if pt_y(vect) < 0:
    #    vect = pt_scale(vect,-1)
    offset = pt1
    return make_ln(offset=offset,vect=vect)
    
def make_ln(offset,vect):
    norm_vect = pt_scale(vect, 1.0 / vect_length(vect))
    return (offset,norm_vect)
    
def horiz_ln(y):
    return make_ln(offset=(0,y),vect=(1,0))
    
def vert_ln(x):
    return make_ln(offset=(x,0),vect=(0,1))
    
def line_offset(ln):
    return ln[0]
    
def line_vector(ln):
    return ln[1]

def normalize(vect):
    return pt_scale(vect,1.0/sqrt(float(dot_prod(vect,vect))))
   
def perpendicular(ln,pt):
    (dx,dy) = line_vector(ln)
    perp_vector = (-1*dy,dx)
    return make_ln(offset=pt,vect=perp_vector)

def projection(pt,ln):
    offset = line_offset(ln)
    pt_o = pt_diff(pt,offset)
    vect = line_vector(ln)
    new_pt = pt_scale(vect,dot_prod(pt_o,vect)/float(dot_prod(vect,vect)))
    return pt_sum(pt_o,offset)

    
def extrapolate(ln,amt):
    #old version
    return pt_sum(line_offset(ln), pt_scale(line_vector(ln),amt))
    return iePt

    
    
def extrapolate_pct(seg,pct):
    (start,end) = end_points(seg)
    
    offset = pt_diff(end,start)
    scaled_offset = pt_scale(offset,pct)
    return pt_sum(start,scaled_offset)
    
def intercept(ln1,ln2):
    (m1,b1) = slope_intercept(ln1)
    
    (m2,b2) = slope_intercept(ln2)
    if m1 == m2:
        return None
    if (m1 == None and b1 == None):
        if (m2 == None and b2 == None):
            return None
        else:
            x = line_offset(ln1)[0]
            y = m2*x + b2
    elif (m2 == None and b1 == None):
        x = line_offset(ln2)[0]
        y = m1*x + b1
    elif(b2 == None or b1 == None or m1 == None or m2 == None):
        return None
    else:
        x = (b2 - b1) / (m1 - m2)
        y = m1*x + b1
    return make_pt(x,y)

        
def make_seg(pt1,pt2):
    ln = make_ln_from_pts(pt1,pt2)
    return (ln[0],ln[1],pt1,pt2)

def end_points(seg):
    return (seg[2],seg[3])

def pt_eq(pt1,pt2):
    (x1,y1) = pt1
    (x2,y2) = pt2
    return feq(x1,x2) and feq(y1,y2)
    
def feq(a,b):
    return abs(a-b) <= 0.001

def seg_contains(seg,pt):
    if len(seg) == 2:
        return True
    if pt_eq(pt,seg[2]) or pt_eq(pt,seg[3]):
        return False
    pt = pt_diff(pt,line_offset(seg))
    t = dot_prod(pt,line_vector(seg))
    t1 = dot_prod(pt_diff(seg[2],line_offset(seg)),line_vector(seg))
    t2 = dot_prod(pt_diff(seg[3],line_offset(seg)),line_vector(seg))
    t_min = min(t1,t2)
    t_max = max(t1,t2)
    if not t_min<t<t_max:
        return False
    return True

def seg_intercept(seg1,seg2):
    inter = intercept(seg1,seg2)
    if not inter:
        return None
    else:
        if not seg_contains(seg1,inter) or not seg_contains(seg2,inter):
            return None
    return inter

def slope_intercept(ln):
    (xo,yo) = line_offset(ln)
    (xv,yv) = line_vector(ln)
    if xv==0:
        return (None,None) #Horizontal lines have no slope_intercept form
    m = float(yv)/xv
    b = yo - float(yv)/xv * xo
    return (m,b)
    
def mirror_pt(pt,ln):
    #First, convert from affine to linear
    offset = line_offset(ln)
    vect = line_vector(ln)
    pt_o = pt_diff(pt,offset)
    pt_r = pt_diff(pt_scale(vect,2 * dot_prod(pt_o,vect) / float(dot_prod(vect,vect))), pt_o)
    return pt_sum(pt_r,offset) 
        
            
def translate_pt(pt,trans):
    (x,y) = pt
    (x_displ,y_displ) = trans
    (x_t,y_t) = (x+x_displ,y+y_displ)
    return (x_t,y_t)

def translate_pts(pts,trans):
    return [translate_pt(pt,trans) for pt in pts]
    
def translate_ln(ln,trans):
    start = extrapolate(ln,-1)
    end = extrapolate(ln,1)
    (new_start,new_end) = translate_pts((start,end),trans)
    return make_ln_from_pts(new_start,new_end)

def rotate_pt(pt,angle,origin=(0,0)):
    (x,y) = pt
    (x_o,y_o) = origin
    (x_n,y_n) = (x-x_o,y-y_o)
    off_rot_x = x_n*cos(angle) - y_n*sin(angle)
    off_rot_y = y_n*cos(angle) + x_n*sin(angle)
    rot_x = off_rot_x + x_o
    rot_y = off_rot_y + y_o
    return (rot_x,rot_y)

def rotate_pts(pts,angle,origin=(0,0)):
    return [rotate_pt(pt,angle,origin) for pt in pts]
    
def rotate_ln(ln,angle,origin=(0.0)):
    start = extrapolate(ln,-1)
    end = extrapolate(ln,1)
    (new_start,new_end) = rotate_pts((start,end),angle,origin)
    return make_ln_from_pts(new_start,new_end)

def scale_pt(pt,amt,origin=(0,0)):
    (x,y) = pt
    (x_o,y_o) = origin
    (x_n,y_n) = (x-x_o,y-y_o)
    (x_ns,y_ns) = (amt*x_n,amt*y_n)
    (x_s,y_s) = (x_ns+x_o,y_ns+y_o)
    return (x_s,y_s)

def scale_pts(pts,amt,origin=(0,0)):
    return [scale_pt(pt,amt,origin) for pt in pts]
    
def scale_ln(ln,amt,origin=(0.0)):
    start = extrapolate(ln,-1)
    end = extrapolate(ln,1)
    (new_start,new_end) = scale_pts((start,end),amt,origin)
    return make_ln_from_pts(new_start,new_end)
    
def angle(pt1,pt2,pt3):
    vect1 = pt_diff(pt1,pt2)
    vect2 = pt_diff(pt3,pt2)
    l1 = vect_length(vect1)
    l2 = vect_length(vect2)
    return arccos(dot_prod(vect1,vect2) / float(l1*l2))
    
def angle_between(v1,v2):
    return arccos(dot_prod(v1,v2) / float(vect_length(v1) * vect_length(v2)))

def rect_center(rect):
    x = 0
    y = 0
    for pt in rect:
        x += pt[0]
        y += pt[1]
    x /= 4
    y /= 4
    return (x,y)
    
def rect_size(rect):
    (bl,br,tr,tl) = rect
    width = br[0] - bl[0]
    height = bl[1] - tl[1]
    return (width,height)
    
def scale_rect(rect,new_width,new_height):
    (bl,br,tr,tl) = rect
    (width,height) = rect_size(rect)
    dx = (width - new_width)/2.0
    dy = (height - new_height)/2.0
    new_bl = (bl[0]+dx,bl[1]-dy)
    new_br = (br[0]-dx,br[1]-dy)
    new_tr = (tr[0]-dx,tr[1]+dy)
    new_tl = (tl[0]+dx,tl[1]+dy)
    return (new_bl,new_br,new_tr,new_tl)
    
def rect_to_cv(rect):
    (bl,br,tr,tl) = rect
    (width,height) = rect_size(rect)
    return (tl[0],tl[1],width,height)
