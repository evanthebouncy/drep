import numpy as np
import math
import matplotlib.pyplot as plot

class CAD:
    def __init__(self, vertices=None):
        self.vertices = vertices or set()

    def make_vertex(self,x,y,epsilon=0.001):
        if any( (xp-x)*(xp-x) + (yp-y)*(yp-y) < epsilon for xp,yp in self.vertices ): return self
        return CAD(self.vertices|{(x,y)})

    def loop(self, objects, transformation, count):
        N = len(objects)
        objects = np.array([[x,y,1.] for x,y in objects]).T

        for _ in range(count):
#            import pdb; pdb.set_trace()
            
            objects = transformation@objects
            for n in range(N):
                self = self.make_vertex(objects[0,n],objects[1,n])
        return self            

    def show(self):
        plot.figure()
        V = list(self.vertices)
        plot.scatter([x for x,_ in V ],[y for _,y in V ])
        plot.show()
        
def mtranslate(x,y):
    return np.array([[1,0,x],
                     [0,1,y],
                     [0,0,1]])

def mrotate(angle, center=None):
    s = math.sin(angle)
    c = math.cos(angle)
    m = np.array([[c,-s,0],
                  [s,c,0],
                  [0,0,1]])
    if center is not None:
        x,y = center
        m = mtranslate(x,y)@m@mtranslate(-x,-y)
    return m

if __name__ == "__main__":
    c = CAD()
    c = c.make_vertex(1,2).make_vertex(1,3)
    c = c.loop([(1,2),(1,3)], mtranslate(2,0.1), 5)
    c = c.loop(c.vertices, mrotate(2*3.14/6,center=(1,2.5)), 5)
    c.show()
