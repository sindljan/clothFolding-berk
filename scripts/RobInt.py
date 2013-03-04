#title           :RobInt.py
#description     :This class is an abstract class called "Robot interface". It defines robot affordences and 
#                 reduce what kind of actions are allowed.
#author          :Jan Sindler
#conact          :sidnljan@fel.cvut.cz
#date            :20130508
#version         :1.0
#usage           :cannot be used alone
#notes           :
#python_version  :2.7.3  
#==============================================================================
#Abstract Robot interface class        
class RobInt:
    
    # manipulation stuff
    def liftUp(self, liftPoint):
        abstract
    def place(self, targPoints):
        abstract
        
    #sensory stuff
    def getImageOfObsObject(self):
        abstract
        
    #graphical stuff
    def get_homography():
        abstract
