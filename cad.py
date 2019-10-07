import numpy as np
import math
import matplotlib.pyplot as plot
import random

class CAD:
    def __init__(self, vertices=None):
        self.vertices = vertices or set()

    def get_size(self, allow_singleton=False):
        if len(self.vertices) <= 1 and not allow_singleton:
            return (0, 0), (1, 1)
        else:
            x_min = min(v[0] for v in self.vertices)
            x_max = max(v[0] for v in self.vertices)
            y_min = min(v[1] for v in self.vertices)
            y_max = max(v[1] for v in self.vertices)
        return (x_min, y_min), (x_max, y_max)

    def bounded_by(self, small, big):
        return all( small <= x <= big and small <= y <= big
                    for x,y in self.vertices )

    def close(self, v1, v2, epsilon=0.001):
        return (v1[0] - v2[0])*(v1[0] - v2[0]) + (v1[1] - v2[1])*(v1[1] - v2[1]) < epsilon*epsilon

    def minimal_spacing(self):
        """Returns the smallest distance in between any two vertices"""
        return min( (v1[0] - v2[0])*(v1[0] - v2[0]) + (v1[1] - v2[1])*(v1[1] - v2[1])
                    for v1 in self.vertices for v2 in self.vertices
                    if v1 != v2)**0.5

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
        def legal(x,y):
            for (xp, yp) in canvas_sofar.vertices:
                if np.sum((np.array([x,y]) - np.array((xp, yp)))**2) < 0.2**2:
                    return False
            return True

        for _ in range(30):
            x, y = np.random.uniform(minmin, maxmax), np.random.uniform(minmin, maxmax)
            if legal(x,y):
                break

        return MakeVertex((x,y))

    def __init__(self, vertex):
        self.vertex = vertex
    def execute(self, cad):
        return cad.make_vertex(*self.vertex)
    def compile(self, spec):
        import cad_repl
        return [cad_repl.Explain(self.vertex)]
    def __str__(self):
        return f"MakeVertex{self.vertex}"

class Loop(Command):
    def __init__(self, selection, mat, repetition):
        self.selection = selection
        self.mat = mat
        self.repetition = repetition
    def execute(self, cad):
        return cad.loop(self.selection, self.mat, self.repetition)

class Translate(Loop):

    @staticmethod
    def sample(canvas, attempts=20):
        for _ in range(attempts):
            focus = random.choice(list(canvas.vertices))
            # select ~80% of all nearby vertices
            neighbors = [v for v in canvas.vertices if canvas.close(v,focus,epsilon=0.4) and v != focus and random.random() < 0.8]

            selection = [focus] + neighbors

            # figure out how big the selection is
            p0,p1 = CAD(set(selection)).get_size(allow_singleton=True)
            w = p1[0] - p0[0]
            h = p1[1] - p0[1]
            # center of selection
            cx = (p1[0] + p0[0])/2
            cy = (p1[1] + p0[1])/2

            # |dx| > w + e
            # n*dx + cx in [0,1]
            e = 0.1 # epsilon
            possibilities = [ (n,dx,dy)
                              for dx in np.linspace(-1,1,30)
                              for dy in np.linspace(-1,1,30)
                              for n in [3,4,5]
                              if abs(dx) > w + e and \
                              abs(dy) > h + e and \
                              0 <= n*dx + cx <= 1 and \
                              0 <= n*dy + cy <= 1]
            
            if len(possibilities) == 0: continue

            n,dx,dy = random.choice(possibilities)

            command = Translate(selection, dx, dy, n)
            new_canvas = command.execute(canvas)

            p0,p1 = new_canvas.get_size()
            if new_canvas.minimal_spacing() > 0.1 and p0[0] >= 0 and p0[1] >= 0 and p1[0] <= 1 and p1[1] <= 1:
                return command
            
        return None

    
    @staticmethod
    def sample_array(canvas, attempts=50):
        for _ in range(attempts):        
            p0 = random.choice(list(canvas.vertices))

            cx,cy = p0
            w,h = 0,0

            # |dx| > w + e
            # n*dx + cx in [0,1]
            e = 0.1 # epsilon
            possibilities = [ (n,dx,dy)
                              for dx in np.linspace(-1,1,30)
                               for dy in np.linspace(-1,1,30)
                               for n in [3,4,5]
                               if abs(dx) > w + e and \
                              abs(dy) > h + e and \
                              0 <= n*dx + cx <= 1 and \
                              0 <= n*dy + cy <= 1]
            if len(possibilities) == 0: continue

            def sort_of_perpendicular(u,v):
                nu = (u*u).sum()**0.5
                nv = (v*v).sum()**0.5
                cosine_of_angle = (u*v).sum()/(nu*nv)
                return abs(cosine_of_angle) < 1./(2**0.5)

            n,dx1,dy1 = random.choice(possibilities)
            remaining_possibilities = [(m,dx2,dy2)
                                       for m,dx2,dy2 in possibilities
                                       if sort_of_perpendicular(np.array([dx1,dy1]),
                                                                np.array([dx2,dy2]))]
            if len(remaining_possibilities) == 0: continue
            
            m,dx2,dy2 = random.choice(remaining_possibilities)

            k1 = Translate({p0},dx1,dy1,n)
            new_canvas = k1.execute(CAD({p0}))
            k2 = Translate(new_canvas.vertices,dx2,dy2,m)
            new_canvas = k2.execute(k1.execute(canvas))
            if new_canvas.minimal_spacing() > 0.1 and new_canvas.bounded_by(0,1):
                return [k1,k2]
        return None
        


    def __init__(self, selection, dx, dy, repetition):
        super(Translate, self).__init__(selection, mtranslate(dx, dy), repetition)
        self.dx, self.dy = dx, dy
    def __str__(self):
        return f"Translate(dx={self.dx}, dy={self.dy}, n={self.repetition}, vs={self.selection})"
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
        if random.random() < 0.75:
            canvas = CAD()
            k0 = MakeVertex.sample(canvas)
            canvas = k0(canvas)
            array = Translate.sample_array(canvas)
            if array:
                return Program([k0] + array)
            
        num_dots = random.randint(1, 3)
        num_loops = random.randint(1,2)
        cmds = []
        canvas_sofar = CAD()

        for i in range(num_dots):
            vertex_cmd = MakeVertex.sample(canvas_sofar)
            canvas_sofar = vertex_cmd(canvas_sofar)
            cmds.append(vertex_cmd)
        
        for j in range(num_loops):

            loop_cmd = Translate.sample(canvas_sofar)
            if loop_cmd is None: continue # can occur if we fail to get a good sample that doesn't lead to overcrowding
            
            canvas_sofar = loop_cmd(canvas_sofar)
            cmds.append(loop_cmd)
        return Program(cmds)

    def __init__(self, commands):
        self.commands = commands

    def __str__(self):
        return "\n".join(map(str, self.commands))

    def __len__(self):
        return len(self.commands)

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
        

    def test4():
        import cad_repl
        for i in range(1000):
            program = Program.sample()
            actions = program.compile()
            crepl = cad_repl.Environment(program.execute())
            crepl.render(f"example_spec_{i}")
            start = crepl
            for index, action in enumerate(actions):
                crepl = action.execute(crepl)
            assert crepl.all_explained

    test4()
        
