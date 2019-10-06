import torch
import torch.nn as nn
from trajectory import Trajectory
from utilities import Module, Softminus
import transformer as tr
from cad_repl import ALL_TAGS, get_trace, Action, TranslateSelect, Death
import cad_repl
import math
import numpy as np
import torch.nn.functional as F
import pprint
pp = pprint.PrettyPrinter(indent=4)

class Agent(Module):

    def __init__(self, num_fourier_components, buttons, hidden_size=64):
        super(Agent, self).__init__()
        num_unique_buttons = sum(a.number_of_points == 1 for a in buttons)
        self.unique_buttons = [a for a in buttons if a.number_of_points == 1]
        self.hidden_size = 64
        self.num_unique_buttons = num_unique_buttons
        self.num_fourier_components = num_fourier_components
        input_vertex_dim = num_fourier_components + len(ALL_TAGS)

        self.embedding = nn.Linear(input_vertex_dim, hidden_size)
        self.transformer = tr.TransformerEncoder(6, 8, hidden_size)

        self._log_value = nn.Sequential(        
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, 1))

        self.unique_btn_fc = nn.Sequential(        
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, self.num_unique_buttons))

        self.selection_fc = nn.Sequential(
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, 1))

        self.meta_head = nn.Linear(hidden_size, 1)
        self.finalize()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)


    def fourier_encoding(self, p, minimum, maximum):
        components = self.num_fourier_components // 2
        x, y = p
        x = (x - minimum[0])/(maximum[0] - minimum[0])
        y = (y - minimum[1])/(maximum[1] - minimum[1])
        waveNumbers = [(kx,ky)
                       for kx in range(int(components**0.5 + 0.6))
                       for ky in range(int(components**0.5 + 0.6))
                       if kx > 0 or ky > 0][:components]
        return [ f(2*math.pi * kx * x + 2*math.pi * ky * y)
                 for kx,ky in waveNumbers
                 for f in [math.sin,math.cos] ]            

    def encode_point(self, x, y, marked_tags, canvas_size):
        tag = [1 if t in marked_tags else 0 for t in ALL_TAGS]
        fourier_point = self.fourier_encoding((x,y), *canvas_size)
        return np.array(tag + fourier_point)

    def encode_environment(self, environment):
        canvas_size = environment.spec.get_size()
        # sort is maybe jank if something is wrong look at here
        return [self.encode_point(x, y, tags, canvas_size)
                for (x,y),tags in sorted(environment.tags.items())]

    def encode_action(self, action):
        raise NotImplementedError

    def forward(self, x):
        """Returns: unique button log probability, selection logs probability, metahead, log value"""
        list_of_enc_points = self.encode_environment(x)
        n_objects = [len(list_of_enc_points)]
        # n_points x enc_point_dim
        x = self.tensor(np.array(list_of_enc_points))
        x = self.embedding(x)
        transformed_xs = self.transformer(x.unsqueeze(0), n_objects).squeeze(0)

        # first output head of selecting 1 btn 1 object
        x_unique_btn = self.unique_btn_fc(transformed_xs)
        x_unique_btn = x_unique_btn.contiguous().view(n_objects[0] * self.num_unique_buttons)
        unique_btn_logprob = F.log_softmax(x_unique_btn, dim=-1).view(n_objects[0], self.num_unique_buttons)

        # second output head of selecting multiple objects (no button here)
        # unormalised logits
        x_selection_logits = self.selection_fc(transformed_xs).squeeze(-1)

        # decide which output to use
        x_meta_head_logit = self.meta_head(torch.max(transformed_xs, dim=-2)[0])

        # decide if we have screwed up
        lv = self._log_value(transformed_xs.sum(-2))
        

        return unique_btn_logprob, x_selection_logits, x_meta_head_logit, lv

    def log_value(self, state):
        return F.logsigmoid(self(state)[-1])

    def loss(self, state=None, action=None, value_target=None):
        """value_target: [0,1], or None, in which case value losses not calculated. Action: action, or None, in which case policy loss is not calculated."""
        unique_logprob, selection_logit, meta_head_logit, log_value = self(state)

        ordered_vertices = list(sorted(list(state.spec.vertices)))

        if value_target is None: value_loss = 0.
        else:
            value_target = self.tensor([value_target])
            value_loss = F.binary_cross_entropy_with_logits(log_value, value_target, reduction='sum')
        if action is not None:
            meta_head_target = self.tensor([1.0 if action.__class__.number_of_points == 1 else 0.0])
            meta_head_loss = F.binary_cross_entropy_with_logits(meta_head_logit, meta_head_target, reduction='sum')
            if action.__class__.number_of_points == 1:
                vertex_index = ordered_vertices.index(action.v)
                action_index = self.unique_buttons.index(action.__class__)
                pred_logprob = unique_logprob[vertex_index][action_index]
                policy_loss = -pred_logprob + meta_head_loss
            else:
                selection_mask = self.tensor([1.0 if v in action.vs else 0.0 for v in ordered_vertices])
                policy_loss = F.binary_cross_entropy_with_logits(selection_logit, selection_mask, reduction='sum') + meta_head_loss
        else:
            policy_loss = 0.

        return policy_loss + value_loss

    def sample_action(self, state):
        ordered_vertices = list(sorted(list(state.spec.vertices)))
        unique_logprob, selection_logit, meta_head_logit, _ = self(state)
        case_unique = self.to_numpy(torch.distributions.bernoulli.Bernoulli(probs=torch.sigmoid(meta_head_logit)).sample()) > 0.5

        if case_unique:
            axis1, axis2 = unique_logprob.size()
            x = self.to_numpy(torch.distributions.categorical.Categorical(probs=torch.exp(unique_logprob).view(-1)).sample())
            return self.unique_buttons[x % axis2](ordered_vertices[x // axis2])
        else:
            x = torch.distributions.bernoulli.Bernoulli(probs=torch.sigmoid(selection_logit)).sample() 
            x = self.to_numpy(x)
            selected_vert = [vert for vert_id, vert in enumerate(ordered_vertices) if x[vert_id] > 0.5]
            return TranslateSelect(set(selected_vert))

    def sample_actions(self, state, n):
        # FIXME: batch
        return [self.sample_action(state) for _ in range(n) ]

    def get_rollout(self, state):
        trajectory = Trajectory(state,[])
        for i in range(20):
            action = self.sample_action(trajectory.final_state)
            try:
                new_state = trajectory.final_state(action)
            except Death:
                new_state = None

            trajectory = trajectory.extend(action, new_state)
            if new_state is None or new_state.all_explained(): break            
            
        return trajectory

    def save(self, loc):
        torch.save(self, loc)

if __name__ == '__main__':

    def test1():
        # MAXIMUM JANK
        agent = Agent(8, Action.all_buttons())
        print ("i didn't die")

        # setting up a spec
        from cad import CAD, Program, MakeVertex, Translate
        from cad_repl import Environment
        c = CAD()
        cmd1 = MakeVertex((1,2))
        cmd2 = MakeVertex((1,3))
        cmd3 = Translate([(1,2),(1,3)], 2, 0.1, 5)
        program = Program([cmd1, cmd2, cmd3])

        trace = get_trace(program)
        pp.pprint(trace)
        for i in range(1000000000000):
            for s, a in trace:
                agent.opt.zero_grad()
                loss = agent.loss(s,a)
                loss.backward()
                agent.opt.step()

            if i % 100 == 0:
                print (loss)
                try:
                    rollout = agent.get_rollout(Environment(program.execute()))
                    print (rollout)
                    print ("we are awesome")
                    assert 0
                except cad_repl.Death:
                    print ("rollout failed")
        assert 0

    test1()


