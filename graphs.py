import matplotlib.pyplot as plot

from utilities import *

def successAtTimeT(results,t):    
    return sum( r[-1].trajectory.final_state.all_explained() for _,r in results) / len(results)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("testResults",nargs='+',default=[])
    arguments = parser.parse_args()

    testResults = [loadPickle(fn) for fn in arguments.testResults ]

    for n,tr in enumerate(testResults):
        plot.figure()
        X = np.arange(0,5,0.1)
        plot.plot(X,[successAtTimeT(tr,x) for x in X ])
        plot.show()
    
    
