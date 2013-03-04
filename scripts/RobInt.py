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
