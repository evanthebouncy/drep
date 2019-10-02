import numpy as np
import math
import matplotlib.pyplot as plot
import random

class CAD:
    def __init__(self, vertices=None):
        self.vertices = vertices or set()

    def get_size(self):
        if len(self.vertices) <= 1:
            return (0, 0), (1, 1)
        else:
            x_min = min(v[0] for v in self.vertices)
            x_max = max(v[0] for v in self.vertices)
            y_min = min(v[1] for v in self.vertices)
            y_max = max(v[1] for v in self.vertices)
        return (x_min, y_min), (x_max, y_max)

    def close(self, v1, v2, epsilon=0.001):
        return (v1[0] - v2[0])*(v1[0] - v2[0]) + (v1[1] - v2[1])*(v1[1] - v2[1]) < epsilon*epsilon

    def closest(self, v, epsilon=0.001):
        """returns the closest vortex within distance epsilon, and returns None if it does not exist"""
        candidates = [vp for vp in self.vertices if self.close(vp,v,epsilon=epsilon) ]
        if candidates: return candidates[0]
        return None        
        
    def make_vertex(self,x,y,epsilon=0.001):
        if any( (xp-x)*(xp-x) + (yp-y)*(yp-y) < epsilon for xp,yp in self.vertices ): return self
        return CAD(self.vertices|{(x,y)})

    def __contains__(self,p):
        return any( self.close(p,pp) for pp in self.vertices  )

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

class Command:
    def __call__(self, other):
        return self.execute(other)
    pass

class MakeVertex(Command):

    @staticmethod
    def sample(canvas_sofar):
        (x_min, y_min), (x_max, y_max) = canvas_sofar.get_size()
        minmin = min(x_min, y_min)
        maxmax = max(x_max, y_max)
        return MakeVertex((np.random.uniform(minmin, maxmax),
                           np.random.uniform(minmin, maxmax)))

    def __init__(self, vertex):
        self.vertex = vertex
    def execute(self, cad):
        return cad.make_vertex(*self.vertex)
    def compile(self, spec):
        import cad_repl
        return [cad_repl.Explain(self.vertex)]

class Loop(Command):
    def __init__(self, selection, mat, repetition):
        self.selection = selection
        self.mat = mat
        self.repetition = repetition
    def execute(self, cad):
        return cad.loop(self.selection, self.mat, self.repetition)

class Translate(Loop):

    @staticmethod
    def sample(canvas_sofar):
        random_number = random.randint(1, len(canvas_sofar.vertices))
        group = random.sample(list(canvas_sofar.vertices), random_number) 
        x_min = min(v[0] for v in group)
        x_max = max(v[0] for v in group)
        y_min = min(v[1] for v in group)
        y_max = max(v[1] for v in group)

        diff_x = x_max - x_min + 1.0
        diff_y = y_max - y_min + 1.0

        delta_x = (diff_x + 1.5 * np.random.random()) * random.choice([1,-1])
        delta_y = (diff_y + 1.5 * np.random.random()) * random.choice([1,-1])

        repetition = np.random.randint(2, 5)

        return Translate(group, delta_x, delta_y, repetition)

    def __init__(self, selection, dx, dy, repetition):
        super(Translate, self).__init__(selection, mtranslate(dx, dy), repetition)
        self.dx, self.dy = dx, dy
    def compile(self, spec):
        import cad_repl
        start_vert = random.choice(self.selection)
        end_vert = spec.closest((start_vert[0] + self.dx, start_vert[1] + self.dy))
        return [
            cad_repl.TranslateStart(start_vert),
            cad_repl.TranslateInduction(end_vert),
            cad_repl.TranslateSelect(self.selection),
        ]

class Program:

    @staticmethod
    def sample():
        num_dots = random.randint(1, 2)
        num_loops = random.randint(1,3)
        cmds = []
        canvas_sofar = CAD()

        for i in range(num_dots):
            vertex_cmd = MakeVertex.sample(canvas_sofar)
            canvas_sofar = vertex_cmd(canvas_sofar)
            cmds.append(vertex_cmd)
        
        for j in range(num_loops):

            vertex_cmd = MakeVertex.sample(canvas_sofar)
            canvas_sofar = vertex_cmd(canvas_sofar)
            cmds.append(vertex_cmd)

            loop_cmd = Translate.sample(canvas_sofar)
            canvas_sofar = loop_cmd(canvas_sofar)
            cmds.append(loop_cmd)
        return Program(cmds)

    def __init__(self, commands):
        self.commands = commands

    def execute(self, cad=None):
        if cad is None:
            cad = CAD()
        for cmd in self.commands:
            cad = cmd(cad)
        return cad

    # turn the program into a list of actions
    # runs the program first to obtain a spec THEN compile (wtf)
    def compile(self):
        spec = self.execute()
        ret = []
        for cmd in self.commands:
            ret.extend(cmd.compile(spec))
        return ret

if __name__ == "__main__":
    def test1():
        c = CAD()
        c = c.make_vertex(1,2).make_vertex(1,3)
        c = c.loop([(1,2),(1,3)], mtranslate(2,0.1), 5)
        c = c.loop(c.vertices, mrotate(2*3.14/6,center=(1,2.5)), 5)
        c.show()
    def test2():
        c = CAD()
        cmd1 = MakeVertex((1,2))
        cmd2 = MakeVertex((1,3))
        cmd3 = Translate([(1,2),(1,3)], 2, 0.1, 15)
        program = Program([cmd1, cmd2, cmd3])
        c_final = program.execute(c)

        actions = program.compile()
        import cad_repl

        crepl = cad_repl.Environment(c_final)
        for index, action in enumerate(actions):
            crepl = action.execute(crepl)
            crepl.render(f"compiled_{index}")

    def test3():
        program = Program.sample()
        actions = program.compile()
        import cad_repl

        crepl = cad_repl.Environment(program.execute())
        for index, action in enumerate(actions):
            crepl = action.execute(crepl)
            crepl.render(f"compiled_{index}")

    test3()
        
