import numpy as np
import math
import matplotlib.pyplot as plot
from matplotlib.lines import Line2D

from cad import CAD, mtranslate, mrotate
from copy import deepcopy

# the agent dies when it executes an invalid command
class Death(Exception): pass

TAG_EXPLAINED = "tag_explained"
TAG_TRANSLATE = "tag_translate"
TAG_TRANSLATE_SELECT = "tag_translate_select"
TAG_TRANSLATE_START = "tag_translate_start"
TAG_TRANSLATE_INDUCTION = "tag_translate_induction"
TAG_ROTATE = "tag_rotate"
TAG_ROTATE_START = "tag_rotate_start"
TAG_ROTATE_NEXT = "tag_rotate_next"
TAG_ROTATE_FINAL = "tag_rotate_final"
TAG_ROTATE_SELECT = "tag_rotate_select"

ALL_TAGS = [
    TAG_EXPLAINED,
    # TAG_ROTATE,
    # TAG_ROTATE_NEXT,
    # TAG_ROTATE_FINAL,
    # TAG_ROTATE_SELECT,
    TAG_TRANSLATE,
    TAG_TRANSLATE_SELECT,
    TAG_TRANSLATE_START,
    TAG_TRANSLATE_INDUCTION
]

class Action: pass

class DoNothing(Action):
    def execute(self,state ): return state

class RotateStart(Action):
    def __init__(self, v):
        self.v = v

    def execute(self, state):
        return state.add_tag_all(TAG_ROTATE).rotate_start(self.v)

class RotateNext(Action):
    def __init__(self, v):
        self.v = v

    def execute(self, state):
        if TAG_ROTATE not in state.tags[self.v]: raise Death()
        state = state.clone()
        state.tags[self.v].add(TAG_ROTATE_NEXT)
        return state

class RotateFinal(Action):
    def __init__(self, v):
        self.v = v

    def execute(self, state):
        if TAG_ROTATE not in state.tags[self.v]: raise Death()
        state = state.clone()
        state.tags[self.v].add(TAG_ROTATE_FINAL)
        return state

class RotateSelect(Action):
    def __init__(self, vs):
        self.vs = vs

    def execute(self, state):
        state = state.clone()
        if len(self.vs) == 0 or TAG_ROTATE not in state.tags[list(self.vs)[0]]: raise Death()

        for v in self.vs: state.tags[v].add(TAG_ROTATE_SELECT)

        v0 = np.array(state.get_tag1(TAG_ROTATE_START))
        v1 = np.array(state.get_tag1(TAG_ROTATE_NEXT))
        v2 = np.array(state.get_tag1(TAG_ROTATE_FINAL))

        # calculate the center
        d01 = ((v0 - v1)*(v0 - v1)).sum()**0.5
        d12 = ((v2 - v1)*(v2 - v1)).sum()**0.5
        
    
class TranslateStart(Action):
    def __init__(self, v):
        self.v = v

    def execute(self, state):
        return state.add_tag_all(TAG_TRANSLATE).translate_start(self.v)

class TranslateInduction(Action):
    def __init__(self, v):
        self.v = v

    def execute(self, state):
        return state.translate_induction(self.v)

class TranslateSelect(Action):
    def __init__(self, objects):
        self.objects = objects

    def execute(self, state):
        return state.translate_select(self.objects)._translate()

class Explain(Action):
    def __init__(self, v):
        self.v = v

    def execute(self, state):
        return state.explain(self.v)

class Environment:

    def __init__(self, spec, tags=None):
        self.spec = spec
        if tags is None:
            self.tags = dict([(x, set()) for x in spec.vertices])
        self.loop_count = dict()

    def render(self, name="repl_render"):
        C = 'bgrcmy'
        radius = 0.1
        plot.figure()
        all_vertices = list(self.spec.vertices)

        for tag_index, TAG in enumerate(ALL_TAGS):
            angle = 2*math.pi*tag_index/len(ALL_TAGS)
            dx,dy = radius*math.cos(angle), radius*math.sin(angle)
            vertex_to_draw = []
            vertex_to_draw_colors = []
            for vertex in all_vertices:
                if TAG in self.tags[vertex]:
                    x,y = vertex
                    vertex_to_draw.append((x + dx, y + dy))
                    vertex_to_draw_colors.append(C[tag_index])
            plot.scatter([x for x,_ in vertex_to_draw],
                         [y for _,y in vertex_to_draw],
                          s=100,
                          c=vertex_to_draw_colors, 
                          alpha=0.3)
        # draw little black dots
        plot.scatter([x for x,_ in all_vertices],
                     [y for _,y in all_vertices],
                      s=10,
                      c='k',
                      alpha=1.0)
        plot.legend(handles=[Line2D([0],[0],marker='o',color='w',label=t,
                                    markerfacecolor=c,markersize=15,alpha=0.3)
                             for c,t in zip(C,ALL_TAGS) ])
        plot.savefig(f"drawings/{name}.png")

    # clone the Environment, if a tags is supplied, use it as the tags
    def clone(self, tags=None):
        if tags is None:
            return deepcopy(self)
        else:
            return Environment(self.spec, tags)

    # remove tag from all vertexes, if it exists
    def remove_tag_all(self, tag):
        for vert in self.tags:
            if tag in self.tags[vert]:
                self.tags[vert].remove(tag)

    # ad. tag from all vertexes
    def add_tag_all(self, tag):
        self = self.clone()
        for vert in self.tags:
            self.tags[vert].add(tag)
        return self

    # ============= DRAWING A SINGLE POINT ==============
    
    # apply a to_explain tag to everything thats not yet explained
    def to_explain(self):
        ret = self.clone()
        for vert in ret.tags:
            if TAG_EXPLAINED not in tags[vert]:
                tags[vert].add(TAG_TO_EXPLAIN)
        return ret

    def close(self, v1, v2, epsilon=0.001):
        return (v1[0] - v2[0])*(v1[0] - v2[0]) + (v1[1] - v2[1])*(v1[1] - v2[1]) < epsilon*epsilon

    def closest(self, v, epsilon=0.001):
        """returns the closest vortex within distance epsilon, and returns None if it does not exist"""
        candidates = [vp for vp in self.tags if self.close(vp,v,epsilon=epsilon) ]
        if candidates: return candidates[0]
        return None        

    # explains a single vertex
    def explain(self, v):
        ret = self.clone()
        for v2,ts in ret.tags.items():
            if self.close(v,v2):
                ts.add(TAG_EXPLAINED)
        return ret

    def get_tag1(self, t):
        """returns the single unique feature which has this tag"""
        matches = [v for v,ts in self.tags.items() if t in ts ]
        if len(matches) != 1: raise Death()
        return matches[0]

    def get_tag(self, t):
        """returns list of all features which have this tag"""
        return [v for v,ts in self.tags.items() if t in ts ]

    # ================== DRAWING A LOOP VIA TRANSLATION ==================

    # select a subset of points to translate
    def translate_select(self, vertices):
        ret = self.clone()
        for vertex in vertices:
            if TAG_EXPLAINED not in self.tags[vertex]: raise Death()
            if TAG_TRANSLATE not in self.tags[vertex]: raise Death()
            ret.tags[vertex].add(TAG_TRANSLATE_SELECT)
        return ret

    # from the set of TAG_TRANSLATE_SELECT points select one to be start point
    # i.e. we pick a special start coordinate u from these points
    def translate_start(self, vertex):
        if TAG_EXPLAINED not in self.tags[vertex]: raise Death()
        if TAG_TRANSLATE not in self.tags[vertex]: raise Death()
        ret = self.clone()
        ret.tags[vertex].add(TAG_TRANSLATE_START)
        return ret

    def translate_induction(self, vertex):
        if TAG_TRANSLATE not in self.tags[vertex]: raise Death()
        ret = self.clone()
        ret.tags[vertex].add(TAG_TRANSLATE_INDUCTION)
        return ret

    # given the subset of points to translate TAG_TRANSLATE_SELECT
    # given a special previledged start coordinate TAG_TRANSLATE_START
    # another vertex will serve as the "step" of the translation, this 
    # step is repeated multiple times until something illegal happens
    def _translate(self):
        ret = self.clone()
        
        # recover the set of selected vertex to be translated
        selected = self.get_tag(TAG_TRANSLATE_SELECT)
    
        # recover the start vertex
        start = self.get_tag1(TAG_TRANSLATE_START)
        step_vertex = self.get_tag1(TAG_TRANSLATE_INDUCTION)

        translate_x = step_vertex[0] - start[0]
        translate_y = step_vertex[1] - start[1]

        # perform the translation
        c = CAD()
        for selected_vertex in selected:
            c = c.make_vertex(*selected_vertex)

        def get_looped_vertices():
            # get the set of looped vertices
            looped_vertices = set()
            for i in range(100):
                # loop it i number of times
                c_looped = c.loop(selected, mtranslate(translate_x, translate_y), i)
                new_vertices = [ret.closest(v) for v in c_looped.vertices ]
                if any( nv is None for nv in new_vertices  ):
                    return looped_vertices, i
                looped_vertices.update(new_vertices)
            assert 0, "should not reach here, how did you loop 100 times and still ok"

        looped_vertices, loop_count = get_looped_vertices()
        for looped_vert in looped_vertices:
            ret.tags[looped_vert].add(TAG_EXPLAINED)
        # do some book keeping
        ret.loop_count[tuple(sorted(list(selected)))] = loop_count

        # clear loop tags
        ret.remove_tag_all(TAG_TRANSLATE_SELECT)
        ret.remove_tag_all(TAG_TRANSLATE_START)
        ret.remove_tag_all(TAG_TRANSLATE_INDUCTION)
        ret.remove_tag_all(TAG_TRANSLATE)

        return ret

if __name__ == "__main__":
    c = CAD()
    c = c.make_vertex(1,2).make_vertex(1,3)
    c = c.loop([(1,2),(1,3)], mtranslate(2,1.001), 5)


    crepl = Environment(c)
    actions = [DoNothing(),
               Explain((1,2)),
               Explain((1,3)),
               TranslateStart((1,2)),
               TranslateInduction((3,3.001)),
               TranslateSelect([(1,2),(1,3)])]
    for index, action in enumerate(actions):
        crepl = action.execute(crepl)
        crepl.render(f"step{index}")
