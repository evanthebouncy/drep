import numpy as np
import math
import matplotlib.pyplot as plot

from cad import CAD, mtranslate, mrotate
from copy import deepcopy

# the agent dies when it executes an invalid command
class Death(Exception): pass

TAG_EXPLAINED = "tag_explained"
TAG_TRANSLATE_SELECT = "tag_translate_select"
TAG_TRANSLATE_START = "tag_translate_start"
TAG_TRANSLATE_INDUCTION = "tag_translate_induction"

ALL_TAGS = [
    TAG_EXPLAINED,
    TAG_TRANSLATE_SELECT,
    TAG_TRANSLATE_START,
    TAG_TRANSLATE_INDUCTION
]

class Environment:

    def __init__(self, spec, tags=None):
        self.spec = spec
        if tags is None:
            self.tags = dict([(x, set()) for x in spec.vertices])
        self.loop_count = dict()

    def render(self, name="repl_render"):
        C = 'bgrcmy'
        plot.figure()
        all_vertices = list(self.spec.vertices)
        # draw untagged
        untagged_vertices = [vertex for vertex in all_vertices if len(self.tags[vertex]) == 0]
        plot.scatter([x for x,_ in untagged_vertices],
                     [y for _,y in untagged_vertices],
                      s=10,
                      c='k',
                      alpha=1.0)
        for TAG in ALL_TAGS:
            vertex_to_draw = []
            vertex_to_draw_colors = []
            for vertex in all_vertices:
                if TAG in self.tags[vertex]:
                    vertex_to_draw.append(vertex)
                    vertex_to_draw_colors.append(C[ALL_TAGS.index(TAG)])
            plot.scatter([x for x,_ in vertex_to_draw],
                         [y for _,y in vertex_to_draw],
                          s=100,
                          c=vertex_to_draw_colors, 
                          alpha=0.3)
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
            ret.tags[vertex].add(TAG_TRANSLATE_SELECT)
        return ret._translate()

    # from the set of TAG_TRANSLATE_SELECT points select one to be start point
    # i.e. we pick a special start coordinate u from these points
    def translate_start(self, vertex):
        if TAG_EXPLAINED not in self.tags[vertex]:
            raise Death()
        ret = self.clone()
        ret.tags[vertex].add(TAG_TRANSLATE_START)
        return ret

    def translate_induction(self, vertex):
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

        return ret

        
if __name__ == "__main__":
    c = CAD()
    c = c.make_vertex(1,2).make_vertex(1,3)
    c = c.loop([(1,2),(1,3)], mtranslate(2,1.001), 5)


    crepl = Environment(c)
    commands = [lambda e: e,
                lambda e: e.explain((1,2)),
                lambda e: e.explain((1,3)),
                lambda e: e.translate_start((1,2)),
                lambda e: e.translate_induction((3,3.001)),
                lambda e: e.translate_select([(1,2),(1,3)])]
    for index, command in enumerate(commands):
        crepl = command(crepl)
        crepl.render(f"step{index}")
