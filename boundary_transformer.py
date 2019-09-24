from transformer import *

import time
import os

class Object():
    def __init__(self, name):
        self.name = name

    @property
    def isVertex(self): return False

    @property
    def isLine(self): return False

    @property
    def extent(self):
        assert False, "extent not implemented. Should return (smallest, largest), each of smallest and largest are vectors."

    def __repr__(self): return str(self)
    def __eq__(self,o): return str(self) == str(o)
    def __ne__(self,o): return not (self == o)
    def __hash__(self): return hash(str(self))

    @staticmethod
    def extract(boundaries, spec):
        os = [Vertex(n, spec, p) for n,p in boundaries.vertices.items() ] + \
             [Line(n, spec, boundaries.vertices[p], boundaries.vertices[q]) for n,(p,q,_) in boundaries.lines.items() ]
        os.sort(key=lambda o: (o.name[0] == 'l', o.name))
        return os
        

class Vertex(Object):
    def __init__(self, name, spec, p):
        super(Vertex, self).__init__(name)
        self.spec = spec
        self.p = p

    @property
    def extent(self):
        return (self.p[0],self.p[1]), (self.p[0],self.p[1])

    @property
    def isVertex(self): return True

    def __str__(self): return f"Vertex(name={self.name}, spec?={self.spec}, pos={self.p})"

class Line(Object):
    def __init__(self, name, spec, p1, p2):
        super(Line, self).__init__(name)
        self.spec = spec
        self.p1 = p1
        self.p2 = p2

    @property
    def extent(self):
        return (min(self.p1[0],self.p2[0]),
                min(self.p1[1],self.p2[1])),\
                (max(self.p1[0],self.p2[0]),
                 max(self.p1[1],self.p2[1]))

    @property
    def isLine(self): return True

    @property
    def isVertical(self): return self.p1[0] == self.p2[0]

    @property
    def isHorizontal(self): return self.p1[1] == self.p2[1]

    def __str__(self): return f"Line(name={self.name}, spec?={self.spec}, p1={self.p1}, p2={self.p2})"
    
class ObjectEncoder(Module):
    def __init__(self, dimensionality):
        super(ObjectEncoder, self).__init__()

        self.dimensionality = dimensionality

        self.finalize()

    def forward(self, objects_batch, normalize=True, fourier=False):
        def encodePosition(components, p, minimum, maximum):
            x = p[0]
            y = p[1]
            if normalize:
                x = (x - minimum[0])/(maximum[0] - minimum[0])
                y = (y - minimum[1])/(maximum[1] - minimum[1])
            waveNumbers = [(kx,ky)
                           for kx in range(int(components**0.5 + 0.6))
                           for ky in range(int(components**0.5 + 0.6)) ][:components]
            return [ f((math.pi/2.) * (2**kx) * x + (math.pi/2.) * (2**ky) * y)
                     for kx,ky in waveNumbers
                     for f in [math.sin,math.cos] ]            
            
        def encodeReal(components, r, minimum, maximum):
            if maximum == minimum:
                p = maximum
                maximum = p + 0.1
                minimum = p - 0.1
            if normalize:
                # Make it live in [0,1]
                r = (r - minimum)/(maximum - minimum)
                assert 0 <= r <= 1

            # First component will have period 1, second will have  2, third will have 4, etc.
            return [ f(2*math.pi*(2**n)*r)
                     for n in range(0, components)
                     for f in [math.sin, math.cos] ]

        if not fourier:
            D = (self.dimensionality - 2)//8
        else:
            D = (self.dimensionality - 2)//4
        def encodeObject(o, minimum, maximum):
            if o.isVertex:
                return [1.,int(o.spec)] + \
                    (encodeReal(D,o.p[0],minimum[0],maximum[0]) + \
                     encodeReal(D,o.p[1],minimum[1],maximum[1]) + \
                     encodeReal(D,o.p[0],minimum[0],maximum[0]) + \
                     encodeReal(D,o.p[1],minimum[1],maximum[1]) if not fourier else \
                     encodePosition(D, o.p, minimum, maximum) + encodePosition(D, o.p, minimum, maximum))
            if o.isLine:
                return [0.,int(o.spec)] + \
                    (encodeReal(D,o.p1[0],minimum[0], maximum[0]) + \
                     encodeReal(D,o.p1[1],minimum[1], maximum[1]) + \
                     encodeReal(D,o.p2[0],minimum[0], maximum[0]) + \
                     encodeReal(D,o.p2[1],minimum[1], maximum[1]) if not fourier else \
                     encodePosition(D, o.p1, minimum, maximum) + encodePosition(D, o.p2, minimum, maximum))
            assert False

        def pad(z): return z + [0.]*(self.dimensionality - len(z))

        output = []
        for os in objects_batch:
            # normalized to [0,1]^2
            extends = [o.extent for o in os]
            minimum = np.array([e[0] for e in extends]).min(0)
            maximum = np.array([e[1] for e in extends]).max(0)
            output.append([pad(encodeObject(o, minimum, maximum)) for o in os])            
            
        return self.tensor(output)

        
class BoundaryTransformer(Transformer):
    def __init__(self):
        D = 128
        vi = ObjectEncoder(D)
        super(BoundaryTransformer, self).__init__(["n","l","RETURN","v","h","start","end",
                                                   "translation"] + [str(i) for i in range(1,10)],
                                                  layers=4,heads=8,
                                                  positional_input=False, positional_output=True,
                                                  vectorizeInput=vi,
                                                  embedding_size=D)
        self._distance = nn.Sequential(nn.Linear(D,D),
                                       nn.ReLU(),
                                       nn.Linear(D,1),
                                       nn.Softplus())
        self.finalize()

    def rollout(self, spec, maximumLength=None):
        maximumLength = maximumLength or 20
        current = SimpleBrep()
        program = []
        specState = Object.extract(spec, True)
        for _ in range(maximumLength):
            canvasState = Object.extract(current, False)
            state = canvasState + specState
            command = self.sample(state, substitutePointers=False)
            print("sampled",command)

            def substituteHorizontal(token):
                if not isinstance(token,Pointer): return token
                if token.index < len(canvasState): return canvasState[token.index].name
                o = state[token.index]
                if o.isVertex: return o.p[0]
                if o.isLine: return o.p1[0]
                assert False
            def substituteVertical(token):
                if not isinstance(token,Pointer): return token
                if token.index < len(canvasState): return canvasState[token.index].name
                o = state[token.index]
                if o.isVertex: return o.p[1]
                if o.isLine: return o.p1[1]
                assert False
                

            # substitution of pointers
            if command[0] == "n":
                command[0] = next_vert_name(current)
                command[1] = substituteHorizontal(command[1])
                command[2] = substituteVertical(command[2])
            elif command[0] == "l":
                command[0] = next_line_name(current)
                command[3] = substituteHorizontal(command[3])
                command[5] = substituteVertical(command[5])
            elif command[0] == "translation":
                command[1] = canvasState[command[1].index].name
                command[2] = canvasState[command[2].index].name
                for command_index in range(4,len(command)):
                    command[command_index] = canvasState[command[command_index].index].name

            command = " ".join(map(str,command))
            print("processes into",command)
            program.append(command)
            try:
                current.execute_command(command)
            except TaoExcept:
                print("Invalid boundary representation command")
                break
            
            if "RETURN" in command: break
        return program, current

class LoopClassifier(Module):
    def __init__(self,fourier=False):
        self.fourier = fourier
        D = 512
        super(LoopClassifier, self).__init__()
        
        self.vectorizeInput = ObjectEncoder(D)
        self._encoder = TransformerEncoder(4,4,D)

        self._classifier = nn.Sequential(nn.Linear(D,3),
                                         nn.LogSoftmax(dim=-1))
        self._loss = nn.NLLLoss(reduction='sum')
        
        self.labelToIndex = {"distractor": 0,
                             "baseCase": 1,
                             "induction": 2}        
        self.finalize()

    def forward(self, state):
        """Takes as input state; returns [logit]"""
        X = self.vectorizeInput([state],fourier=self.fourier)
        e = self._encoder(X, [ len(state) ])
        return self._classifier(e.squeeze(0))

    def loss(self, state, groundTruth):
        prediction = self(state)
        assert all( gt in self.labelToIndex for gt in groundTruth )
        y = self.tensor([self.labelToIndex[gt] for gt in groundTruth])
        return self._loss(prediction,y)

    def accuracy(self, state, groundTruth):
        yh = self.predict(state)
        assert all( gt in self.labelToIndex for gt in groundTruth )
        return [yh[n,i].cpu().data.numpy()
                for n,i in enumerate([self.labelToIndex[gt] for gt in groundTruth]) ]
        

    def predict(self, state):
        return self(state).exp()
        
def makeLoopExample(trace):
    # assume that there is exactly one loop
    assert sum('translation' in command for command,_ in trace ) == 1
    
    finalState = Object.extract(trace[-1][1], True)
    translationIndex = [i for i,(command,_) in enumerate(trace) if 'translation' in command][0]

    baseCase = set(Object.extract(trace[translationIndex - 2][1], True))
    induction = set(Object.extract(trace[translationIndex][1], True)) - baseCase
    distractors = set(finalState) - baseCase - induction

    labels = []
    for o in finalState:
        if o in distractors:
            labels.append("distractor")
        elif o in baseCase:
            labels.append("baseCase")
        elif o in induction:
            labels.append("induction")
        else:
            assert False
            
    return finalState, labels
    
    
def trainClassifier(maximum_size, fourier=False):
    start_time = time.time()
    m = LoopClassifier(fourier=fourier)

    optimizer = torch.optim.Adam(m.parameters(), lr=0.001, eps=1e-3, amsgrad=True)

    totalLosses = []
    last_update = time.time()
    
    while True:
        tr = random_classifier_trace(bodySize=random.choice(range(1,maximum_size+1)),
                                     distractors=random.choice(range(0,maximum_size+1)))
            
        state, target = makeLoopExample(tr)

        m.zero_grad()
        L = m.loss(state, target)
        L.backward()
        optimizer.step()

        totalLosses.append(L.data.cpu().numpy())

        if time.time() - last_update > 5:#30*60: # every half hour we give a loss update
            print(f"Average loss {sum(totalLosses)/len(totalLosses)}\t{len(totalLosses)/(time.time()-last_update)} gradient steps/sec")
            os.system("mkdir  -p checkpoints")
            torch.save(m, f"checkpoints/classifier_fourier={fourier}.p")
            last_update = time.time()
            totalLosses = []
def testClassifier(fourier=False, maximum_size=3):
    import matplotlib.pyplot as plot
    
    m = load_checkpoint(f"checkpoints/classifier_fourier={fourier}.p")
    directory = "data/classify/"
    if fourier: directory += "fourier"
    else: directory += "no_fourier"
    os.system(f"mkdir  -p {directory} && rm {directory}/*")

    random.seed(42)
    testCases = []
    while len(testCases) < 50:
        tr = random_classifier_trace(bodySize=random.choice(range(1,maximum_size+1)),
                                     distractors=random.choice(range(0,maximum_size+1)))
        # make sure it has a translation in it
        if sum('translation' in command for command,_ in tr ) == 1:
            testCases.append(tr)
    accuracy = []
    for n,tr in enumerate(testCases):
        state, target = makeLoopExample(tr)
        predictions = m.predict(state).cpu().data.numpy()
        accuracy.extend(m.accuracy(state, target))

        plot.figure()
        plot.scatter([ v.p[0] for v in state ],
                     [ v.p[1] for v in state ],
                     c=predictions)
        plot.savefig(f"{directory}/{n}_prediction.png")
        plot.close()
        plot.figure()

        indexToColor = {"distractor": (1,0,0.),
                        "baseCase": (0.,1.,0.),
                        "induction": (0.,0.,1.)}
        plot.scatter([ v.p[0] for v in state ],
                     [ v.p[1] for v in state ],
                     c=[ indexToColor[t]
                         for t in target ])
        plot.savefig(f"{directory}/{n}_groundTruth.png")
        plot.close()

    print(f"Average accuracy: {sum(accuracy)/len(accuracy)}")
        
        
        
        
def makeTrainingExamples(trace):
    """trace should be a [(command,new_state)]. returns a list of (state, command)"""
    current = SimpleBrep()
    spec = trace[-1][1]
    examples = []
    for command, new_state in trace + [("RETURN",spec)]:
        state = Object.extract(current, False) + Object.extract(spec, True)
        def horizontalPointer(x):
            try:
                x = float(x)
                pi = None
                for objectIndex, o in enumerate(state):
                    if o.isVertex and o.p[0] == x or \
                       o.isLine and o.isVertical and o.p1[0] == x:
                        pi = objectIndex
                        break
                if pi is None:
                    print("FATAL")
                    print(x)
                    print(state)
                assert pi is not None
                return Pointer(pi)
            except (ValueError, TypeError): return x
        def verticalPointer(y):
            try:
                y = float(y)
                pi = None
                for objectIndex, o in enumerate(state):
                    if o.isVertex and o.p[1] == y or \
                       o.isLine and o.isHorizontal and o.p1[1] == y:
                        pi = objectIndex
                        break
                if pi is None:
                    print("FATAL")
                    print(y)
                    print(state)
                assert pi is not None
                return Pointer(pi)
            except (ValueError, TypeError): return y

        tokens = command.split(" ")
        if tokens[0] == "RETURN":
            command = ["RETURN"]
        elif tokens[0] == "translation":
            command = tokens[1:]
            command = [ token if token[0] not in ['n','l'] else \
                        Pointer([pi for pi,obj in enumerate(state) \
                                 if obj.name == token and not obj.spec][0]) 
                        for token in command ]
            command = ["translation"] + command
        else:
            command = tokens[1:]
            # convert the command into a list of symbols and pointers
            # First convert pointers
            command = [ token if token[0] not in ['n','l'] else \
                        Pointer([pi for pi,obj in enumerate(state) if obj.name == token][0]) 
                        for token in command ]
            if tokens[0][0] == 'n': # vertex
                command[0] = horizontalPointer(command[0])
                command[1] = verticalPointer(command[1])
            elif tokens[0][0] == 'l':
                pass # lines never use absolute coordinates
            else:
                assert False
            command = [tokens[0][0]] + command

        examples.append((state, command))
        current = new_state
    if False:
        print(trace)
        print("EXAMPLES")
        for state, command in examples:
            print("in the state",state)
            print("execute",command)
    
    return examples

def imitation(maximum_size=10):
    if False and os.path.exists("checkpoints/transformer.p"):
        bt = load_checkpoint("checkpoints/transformer.p")
    else:
        bt = BoundaryTransformer()
    optimizer = torch.optim.Adam(bt.parameters(), lr=0.001, eps=1e-3, amsgrad=True)
    totalLosses = []
    last_update = time.time()

    last_demo = time.time()

    def demo():
        print("DEMO")
        with torch.no_grad():
            for _ in range(20):
                boundaries = random_trace(size=maximum_size)[-1][1]
                try:
                    r = bt.rollout(boundaries)        
                    print("Spec vertices:", boundaries.vertices, "\nRollout vertices", r[1].vertices)
                    print("Spec lines:", boundaries.lines, "\nRollout lines", r[1].lines)
                    print("Rollout program:")
                    print("\n".join(r[0]))
                except: print("some kind of exception")
        print()
    
    while True:
        for state, command in makeTrainingExamples(random_trace(size=maximum_size)):
            bt.zero_grad()
            l = -bt.logLikelihood(state, command)
            l.backward()
            optimizer.step()
            totalLosses.append(l.data.cpu().numpy())
            if time.time() - last_update > 30*60: # every half hour we give a loss update
                os.system("mkdir  -p checkpoints")
                torch.save(bt, "checkpoints/transformer.p")
                print(f"Average loss {sum(totalLosses)/len(totalLosses)}\t{len(totalLosses)/(time.time()-last_update)} gradient steps/sec")
                last_update = time.time()
                totalLosses = []

            if time.time() - last_demo > 60*60: # every hour we give a demo
                demo()
                last_demo = time.time()
    
            
def test(spec, maximumLength=None):        
    
    best_solution = None
    def distance(b1,b2):
        b1 = Object.extract(b1,b1)
        b2 = Object.extract(b2,b2)
        return \
            len({ (v.p[0],v.p[1]) for v in b1 if v.isVertex} ^ { (v.p[0],v.p[1]) for v in b2 if v.isVertex}) + \
            len({ tuple(sorted(((v.p2[0],v.p2[1]),(v.p1[0],v.p1[1])))) for v in b1 if not v.isVertex} ^ { tuple(sorted(((v.p2[0],v.p2[1]),(v.p1[0],v.p1[1])))) for v in b2 if not v.isVertex})
    
    for _ in range(10):
        try:
            r = bt.rollout(boundaries, maximumLength=maximumLength)
            b = r[1]
            if best_solution is None or distance(best_solution[1],spec) > distance(b,spec):
                best_solution = r
        except: continue
    if best_solution is not None: print("DISTANCE",distance(best_solution[1],spec))
    return best_solution

if __name__ == "__main__":
    from random_gen import *

    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("mode",type=str,
                        choices=["imitation","test","demo","classifier",
                                 "classify"])
    parser.add_argument("--fourier", action='store_true',
                        default=False)
    parser.add_argument("--maxObjects", type=int,
                        default=10)
    arguments = parser.parse_args()

    print("cuda?", torch.cuda.is_available())

    if arguments.mode == "imitation":
        imitation(maximum_size=arguments.maxObjects)
    elif arguments.mode == "classify":
        testClassifier(fourier=arguments.fourier, maximum_size=arguments.maxObjects)
    elif arguments.mode == "classifier":
        trainClassifier(maximum_size=arguments.maxObjects,
        fourier=arguments.fourier)
    elif arguments.mode == "demo":
        os.system("rm -r demo/*&&mkdir demo")
        for n in range(10):
            trace = random_trace(size=arguments.maxObjects)
            display_brep(trace[-1][1], f"demo/{n}.png")
            for ci, (state, command) in enumerate(makeTrainingExamples(trace)):
                display(trace[min(ci,len(trace)-1)][1], trace[-1][1], f"demo/{n}_{ci}.png", title=command)
    elif arguments.mode == "test":
        os.system("mkdir data && rm data/*")
        bt = load_checkpoint("checkpoints/transformer.p")
        with torch.no_grad():
            for ti in range(20):
                tr = random_trace(size=arguments.maxObjects)
                program = "\n".join(tr_[0] for tr_ in tr)
                boundaries = tr[-1][-1]
                print()
                print("Test case number",ti,"corresponding to the program:")
                print(program)
                r = test(boundaries, maximumLength=len(tr)+1)
                if r is None: continue
                
                print("Spec vertices:", boundaries.vertices, "\nRollout vertices", r[1].vertices)
                print("Spec lines:", boundaries.lines, "\nRollout lines", r[1].lines)
                print("Rollout program:")
                print("\n".join(r[0]))

                canvas = SimpleBrep()
                for n in range(len(r[0])):
                    if "RETURN" in r[0][n]: continue
                    try:
                        canvas.execute_command(r[0][n])
                    except TaoExcept: continue
                    
                    display(canvas, boundaries, f"data/{ti}_{n}.png",
                            title=r[0][n])
        print()


