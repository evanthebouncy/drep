"""reinforcement learning algorithms"""

import torch

def train_policy_and_value(agent, _=None,
                           sample_problem=None, # should sample a pair of a reward function (which maps trajectories to {0,1}) and a list of state-action pairs
):
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
    while True:
        reward_function, imitation_trace = sample_problem()

        with torch.no_grad():
            trajectory = agent.get_rollout(imitation_trace[0][0])

        R = reward_function(trajectory)
        optimizer.zero_grad()
        loss = 0
        if R > 0.9:
            # reinforce our actual actions, and supervise the value on having succeeded
            for state, action in trajectory.state_actions:
                loss = loss + agent.loss(state, action, R)
        else:
            # reinforce the teacher 's actions and supervise the value on having failed
            for state, action in imitation_trace:
                try:
                    loss = loss + agent.loss(state, action)
                except:
                    print(f"WARNING: Got exception when calculating loss for state/action:")
                    print(state)
                    print(action)
            for state, action in trajectory.state_actions:
                try:
                    loss = loss + agent.loss(state, None, R)
                except:
                    print(f"WARNING: Got exception when calculating loss for state/action:")
                    print(state)
                    print(action)                

        loss.backward()
        optimizer.step()

        yield agent.to_numpy(loss)
        
