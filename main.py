from cad import *
from SMC import *
from cad_repl import *
from agent import *
from rl import *
from utilities import *

import sys

# Evan my container does not have this package I am so sorry
try:
    from tqdm import tqdm
except:
    def tqdm(g): return g

def imitation_train(agent, checkpoint):
    all_losses = []
    for i in tqdm(range(1000000000000)):
        # sample and train
        program = Program.sample()
        try:
            trace = get_trace(program)
        except:
            print("FYI, the following program give us some trouble. It's probably not a big deal.")
            print(program)
            continue
        

        for s, a in trace:                                   
            try:
                agent.opt.zero_grad()
                loss = agent.loss(s,a)
                loss.backward()
                agent.opt.step()
                all_losses.append(agent.to_numpy(loss))
            except ValueError:
                print ("well we dont care")

        # once in awhile give some prints
        if len(all_losses) > 1000:
            print (f"average loss {np.mean(all_losses)}")
            all_losses = []
            rollout = agent.get_rollout(Environment(program.execute()))
            if rollout is not None:
                print("Rollout:")
                for s,a in rollout.state_actions:
                    print(a)
                if rollout.final_state is None:
                    print("then we die")
                else:
                    print(f"Final state: All explained? {rollout.final_state.all_explained()}")
            agent.save(checkpoint)

def do_reinforcement_learning(agent, checkpoint):
    def sample_problem():
        while True:
            try:
                program = Program.sample()
                trace = get_trace(program)
            except:
                continue

            def R(trajectory):
                # We get a reward if the following conditions all hold:
                # 1. do not die
                # 2. explain everything
                # 3. your program has to be at least as short as the ground truth program
                if trajectory.final_state is None: return 0.
                if not trajectory.final_state.all_explained(): return 0.
                sampled_program = trajectory.to_program()
                if len(sampled_program) <= len(program): return 1.
                else: return 0.

            return R, trace                
        
    losses = []
    for loss in train_policy_and_value(agent, sample_problem=sample_problem):
        losses.append(loss)
        if len(losses) > 500:
            print(f"Average loss {sum(losses)/len(losses)}")
            losses = []
            agent.save(checkpoint)
        

def test(loc, n_test, timeout):
    agent = torch.load(loc)

    # sample a bunch of programs.
    test_programs = [Program.sample() for _ in range(n_test)]
    # for each program we are going to store the best trajectory that solves it
    best_trajectories = []

    # count how many of the problems we solve
    solved = 0
    for program in test_programs:
        sampler = SMC(agent, maximumLength=len(get_trace(program)), valueCoefficient=1., initialParticles=10, exponentialGrowthFactor=2)
        initial_state = Environment(program.execute())

        # keep track of the sample which explains the largest fraction of the data
        best_trajectory = None

        # loop over the samples and maintain the one which is best
        for trajectory in sampler.infer(initial_state, timeout=timeout):
            if best_trajectory is None or \
               best_trajectory.final_state.number_explained() <= trajectory.final_state.number_explained():
                best_trajectory = trajectory

        # check to see if we solved the problem
        if best_trajectory is not None and best_trajectory.final_state.all_explained():
            solved += 1
        
        best_trajectories.append(best_trajectory)
    
    print("using SMC we solve", solved/len(test_programs), "optimally within", timeout, "seconds.")

    # visualize the solutions
    for n, rollout in enumerate(best_trajectories):
        if rollout is None: continue
        for i,(s,a) in enumerate(rollout.state_actions + [(rollout.final_state,"DONE")]):
            s.render(f"problem_{n}_step_{i}", title=str(a))
        print("Program #",n)
        print(rollout.to_program())
            
if __name__ == '__main__':
    # say what u want to do
    import argparse
    parser = argparse.ArgumentParser(description = "")

    parser.add_argument("mode",type=str,
                        choices=["imitation", "rl", "test", "demo"])
    parser.add_argument("--checkpoint", type=str,
                        default="saved_models/m1.mdl")
    parser.add_argument("--export", default="saved_models/m2.mdl")
    parser.add_argument("--numtest","-n",
                        type=int,
                        default=10)
    parser.add_argument("--timeout","-t",
                        type=float,
                        default=5)
    parser.add_argument("--hidden","-H",
                        type=int,
                        default=128,
                        help="hidden layer size")

    arguments = parser.parse_args()

    # actually do shit
    if arguments.mode == 'imitation':
        agent = Agent(16, Action.all_buttons(), hidden_size=arguments.hidden)
        imitation_train(agent, arguments.checkpoint)
    if arguments.mode == "rl":
        do_reinforcement_learning(torch.load(arguments.checkpoint),
                                  arguments.export)
    if arguments.mode == 'test':
        test(arguments.checkpoint, arguments.numtest, arguments.timeout)
    if arguments.mode == 'demo':
        for n in range(20):
            p = Program.sample()
            Environment(p.execute()).render(f"demo_{n}")


