#Abstract Robot interface class        
class RobInt:
    def liftUp(self, liftPoint):
        abstract
    def place(self, targPoints):
        abstract
    def getImageOfObsObject(self):
        abstract
