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

        print ([x[1] for x in trace])
        print (program)
        for s, a in trace:                                   
            agent.opt.zero_grad()
            loss = agent.loss(s,a)
            loss.backward()
            agent.opt.step()

            all_losses.append(agent.to_numpy(loss))

        # once in awhile give some prints
        if len(all_losses) > 1000:
            print (f"average loss {np.mean(all_losses)}")
            all_losses = []
            for i, (state, _) in enumerate(trace):
                state.render(f"compiled_{i}")
            try:                                                            
                rollout = agent.get_rollout(Environment(program.execute()))
                print (rollout)
                print ("we are awesome")                        
            except cad_repl.Death:                                                  
                print ("rollout failed")


if __name__ == '__main__':
    agent = Agent(8, Action.all_buttons())
    train(agent)


