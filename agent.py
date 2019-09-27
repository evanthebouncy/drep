import torch
import torch.nn as nn
from utilities import Module
import transformer as tr
from cad_repl import ALL_TAGS
import math
import numpy as np

class Agent(Module):

    def __init__(self, num_fourier_components, hidden_size=64):
        super(Agent, self).__init__()
        self.hidden_size = 64
        self.num_fourier_components = num_fourier_components
        input_vertex_dim = num_fourier_components + len(ALL_TAGS)

        self.embedding = nn.Linear(input_vertex_dim, hidden_size)
        self.transformer = tr.TransformerEncoder(6, 8, 64)

        self.finalize()

    def fourier_encoding(self, p, minimum, maximum):
        components = self.num_fourier_components // 2
        x, y = p
        x = (x - minimum[0])/(maximum[0] - minimum[0])
        y = (y - minimum[1])/(maximum[1] - minimum[1])
        waveNumbers = [(kx,ky)
                       for kx in range(int(components**0.5 + 0.6))
                       for ky in range(int(components**0.5 + 0.6)) ][:components]
        return [ f((math.pi/2.) * (2**kx) * x + (math.pi/2.) * (2**ky) * y)
                 for kx,ky in waveNumbers
                 for f in [math.sin,math.cos] ]            

    def encode_point(self, x, y, marked_tags, canvas_size):
        tag = [1 if t in marked_tags else 0 for t in ALL_TAGS]
        fourier_point = self.fourier_encoding((x,y), *canvas_size)
        return np.array(tag + fourier_point)

    def encode_environment(self, environment):
        canvas_size = environment.spec.get_size()
        return [self.encode_point(x, y, tags, canvas_size)
                for (x,y),tags in environment.tags.items()]

if __name__ == '__main__':

    def test1():
        agent = Agent(8)
        print ("i didn't die")

        from cad import CAD, Program, MakeVertex, Translate
        c = CAD()
        cmd1 = MakeVertex((1,2))
        cmd2 = MakeVertex((1,3))
        cmd3 = Translate([(1,2),(1,3)], 2, 0.1, 15)
        program = Program([cmd1, cmd2, cmd3])
        c_final = program.execute(c)
        actions = program.compile()
        import cad_repl
        crepl = cad_repl.Environment(c_final)

        hi = agent.encode_environment(crepl)
        print (hi)

    test1()


