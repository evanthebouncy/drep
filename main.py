from cad import *
from cad_repl import *
from agent import *
import tqdm

def train(agent):
    all_losses = []
    for i in tqdm.tqdm(range(1000000000000)):
        # sample and train
        try:
            program = Program.sample()
            trace = get_trace(program)
        except:
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
            try:                                                            
                rollout = agent.get_rollout(Environment(program.execute()))
                print (rollout)
                if rollout is not None:
                    print ("we are awesome")
            except cad_repl.Death:                                                  
                print ("rollout failed")
            agent.save("saved_models/m1.mdl")

def test(loc, n_test):
    agent = torch.load(loc)

    test_progs = [Program.sample() for _ in range(n_test)]
    rollouts = [agent.get_rollout(Environment(program.execute())) for program in test_progs]
    success = sum([final_state.all_explained() for _, final_state in rollouts])

    print (success / n_test)

    for i, (rollout,final_state) in enumerate(rollouts):
        for j, crepl in enumerate([x[0] for x in rollout] + [final_state]):
            crepl.render(f"problem_{i}_step_{j}")

if __name__ == '__main__':
    # say what u want to do
    import argparse
    parser = argparse.ArgumentParser(description = "")

    parser.add_argument("mode",type=str,
                        choices=["train", "test"])
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--numtest",type=int,
                        default=10)

    arguments = parser.parse_args()

    # actually do shit
    if arguments.mode == 'train':
        agent = Agent(8, Action.all_buttons())
        train(agent)
    if arguments.mode == 'test':
        test(arguments.checkpoint, arguments.numtest)


