import numpy as np
import math
import matplotlib.pyplot as plot

from cad import CAD, mtranslate, mrotate
from copy import deepcopy

TAG_TO_EXPLAIN = "tag_to_explain"
TAG_EXPLAINED = "tag_explained"
TAG_TRANSLATE_SELECT = "tag_translate_select"
TAG_TRANSLATE_START = "tag_translate_start"

ALL_TAGS = [
        TAG_TO_EXPLAIN,
        TAG_EXPLAINED,
        TAG_TRANSLATE_SELECT,
        TAG_TRANSLATE_START,
        ]

class Crepl:

    def __init__(self, spec, tags=None):
        self.spec = spec
        if tags is None:
            self.tags = dict([(x, set()) for x in spec.vertices])
        self.loop_count = dict()

    def render_png(self, name="repl_render"):
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

    # clone the Crepl, if a tags is supplied, use it as the tags
    def clone(self, tags=None):
        if tags is None:
            return deepcopy(self)
        else:
            return Crepl(self.spec, tags)

    # remove tag from all vertexes, if it exists
    def remove_tag_all(self, tag):
        for vert in self.tags:
            if tag in self.tags[vertex]:
                self.tags[vert].remove(tag)

    # ============= DRAWING A SINGLE POINT ==============
    
    # apply a to_explain tag to everything thats not yet explained
    def to_explain(self):
        ret = self.clone()
        for vert in ret.tags:
            if TAG_EXPLAINED not in tags[vert]:
                tags[vert].add(TAG_TO_EXPLAIN)
        return ret

    # from the set of TAG_TO_EXPLAIN points select a subset to explain
    # also removes the "to_explain" tag
    def explain(self, vertices):
        ret = self.clone()
        for vert in vertices:
            assert TAG_TO_EXPLAIN in ret.tags[vert]
        ret.remove_tag_all(TAG_TO_EXPLAIN)
        for vert in vertices:
            ret.tags[vert].add(TAG_EXPLAINED)
        return ret

    # ================== DRAWING A LOOP VIA TRANSLATION ==================

    # select a subset of points to translate
    def translate_select(self, vertices):
        ret = self.clone()
        for vert in vertices:
            ret.tags[vert].add(TAG_TRANSLATE_SELECT)
        return ret

    # from the set of TAG_TRANSLATE_SELECT points select one to be start point
    # i.e. we pick a special start coordinate u from these points
    def translate_start(self, vertex):
        ret = self.clone()
        assert TAG_TRANSLATE_SELECT in ret.tags[vertex]
        ret.tags[vertex].add(TAG_TRANSLATE_START)
        return ret

    # given the subset of points to translate TAG_TRANSLATE_SELECT
    # given a special previledged start coordinate TAG_TRANSLATE_START
    # another vertex will serve as the "step" of the translation, this 
    # step is repeated multiple times until something illegal happens
    def translate_step(self, step_vertex):
        ret = self.clone()
        
        # recover the set of selected vertex to be translated
        selected = set()
        for vert in ret.tags:
            if TAG_TRANSLATE_SELECT in ret.tags[vert]:
                selected.add(vert)
        assert len(selected) > 0, "this si wrong l o l"
    
        # recover the start vertex
        start = None
        for vert in ret.tags:
            if TAG_TRANSLATE_START in ret.tags[vert]:
                start = vert
        assert start is not None, "cmon man this is wrong l o l"

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
                print (selected)
                print (i)
                print (translate_x, translate_y)
                c_looped = c.loop(selected, mtranslate(translate_x, translate_y), i)
                print (c_looped.vertices)
                for loop_vert in c_looped.vertices:
                    # break out of the loop if the loop goes outside the spec
                    if loop_vert not in ret.spec.vertices:
                        return looped_vertices, i
                looped_vertices = looped_vertices | c_looped.vertices
            assert 0, "should not reach here, how did you loop 100 times and still ok"

        looped_vertices, loop_count = get_looped_vertices()
        for looped_vert in looped_vertices:
            ret.tags[looped_vert] = TAG_EXPLAINED
        # do some book keeping
        ret.loop_count[tuple(sorted(list(selected)))] = loop_count

        return ret

        
if __name__ == "__main__":
    c = CAD()
    c = c.make_vertex(1,2).make_vertex(1,3)
    c = c.loop([(1,2),(1,3)], mtranslate(2,1), 5)


    crepl = Crepl(c)
    crepl.render_png("step0")

    crepl = crepl.translate_select([(1,2),(1,3)])
    print (crepl.tags)
    crepl.render_png("step1")

    crepl = crepl.translate_start((1,2))
    print (crepl.tags)
    crepl.render_png("step2")

    crepl = crepl.translate_step((3,3))
    print (crepl.tags)
    crepl.render_png("step3")



