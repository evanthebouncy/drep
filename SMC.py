import torch

from test_result import *
from trajectory import Trajectory
from cad_repl import Death

import numpy as np

import math
import time


class SMC:
    def __init__(self, agent, _=None,
                 maximumLength=8,
                 initialParticles=20, exponentialGrowthFactor=2,
                 valueCoefficient=1.):
        self.maximumLength = maximumLength
        self.initialParticles = initialParticles
        self.exponentialGrowthFactor = exponentialGrowthFactor
        self.valueCoefficient = valueCoefficient
        self.agent = agent

    def inferTestResults(self, initialState, timeout, reward):
        results = []
        start = time.time()
        for trajectory in self.infer(initialState, timeout):
            T = time.time() - start
            R = reward(trajectory)
            if len(results) == 0 or results[-1].reward < R:
                results.append(TestResult(trajectory, T, R))
        return results            
        
    def infer(self, initialState, timeout):
        """Yields a stream of Trajectory's until timeout is reached"""
        
        startTime = time.time()
        numberOfParticles = self.initialParticles
        
        class Particle():
            def __init__(self, state, frequency, trajectories):
                # trajectories is a map from trajectory to frequency
                # intuitively it is a histogram of the different trajectories that lead to this final state
                # or you can think of it as a multiset of trajectories
                self._trajectories = list(trajectories.items())
                self.frequency = frequency
                self.state = state
                self.log_value = None

                assert frequency == sum(trajectories.values())
                
            def __str__(self):
                return f"Particle(frequency={self.frequency}, logV={self.log_value}, state={self.state})"
            @property
            def immutableCode(self):
                return self.state
                
            def __eq__(self,o):
                return self.immutableCode == o.immutableCode
            def __ne__(self,o): return not (self == o)
            def __hash__(self): return hash(self.immutableCode)

            def trajectories(self):
                """Generates a sequence of trajectories"""
                for trajectory, frequency in self._trajectories:
                    for _ in range(frequency):
                        yield trajectory

            def resample_trajectories(self):
                ps = [frequency for _, frequency in self._trajectories ]
                z = sum(ps)
                ps = [p/z for p in ps ]
                samples = np.random.multinomial(self.frequency, ps)
                self._trajectories = {tr: f for (tr,_),f in zip(self._trajectories,samples)
                                      if f > 0 }
                self._trajectories = list(self._trajectories.items())

        while time.time() - startTime < timeout:
            population = [Particle(initialState, numberOfParticles,
                                   {Trajectory(initialState, []): numberOfParticles})]

            for generation in range(self.maximumLength):
                if time.time() - startTime > timeout: break
                
                sampleFrequency = {} # map from [state][trajectory] to frequency
                for p in population:
                    for action, trajectory in zip(self.agent.sample_actions(p.state, p.frequency),
                                                  p.trajectories()):
                        try:
                            new_state = action(p.state)
                        except Death:
                            continue
                        sampleFrequency[new_state] = sampleFrequency.get(new_state, {})
                        # add-on to the trajectory the fact that we took this action in this state
                        new_trajectory = trajectory.extend(action, new_state)
                        sampleFrequency[new_state][new_trajectory] = sampleFrequency[new_state].get(new_trajectory, 0) + 1

                for new_state, trajectories in sampleFrequency.items():
                    for tr in trajectories:
                        yield tr
                
                # Convert states to particles
                samples = [Particle(state, sum(trajectories.values()), trajectories)
                           for state, trajectories in sampleFrequency.items() ]
                
                # Computed value
                for p in samples:
                    p.log_value = self.agent.log_value(p.state).cpu().data.item()

                # Resample
                logWeights = [math.log(p.frequency) + p.log_value*self.valueCoefficient
                              for p in samples ]
                ps = [ math.exp(lw - max(logWeights)) for lw in logWeights ]
                ps = [p/sum(ps) for p in ps]
                sampleFrequencies = np.random.multinomial(numberOfParticles, ps)

                population = []
                for particle, frequency in sorted(zip(samples, sampleFrequencies),
                                                  key=lambda sf: sf[1]):
                    particle.frequency = frequency
                    if frequency > 0:
                        particle.frequency = frequency
                        # resample trajectories, because we have adjusted the frequency of this particle
                        particle.resample_trajectories()
                        population.append(particle)
                        
                if len(population) == 0: break
                
            numberOfParticles *= self.exponentialGrowthFactor
            print("Increased number of particles to",numberOfParticles)
