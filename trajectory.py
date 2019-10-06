
class Trajectory:
    """Wrapper class for bundling up a list of (state, action) as well as the final resulting state"""
    def __init__(self, final_state, state_actions):
        self.final_state = final_state
        self.state_actions = state_actions

    def extend(self, action, state):
        return Trajectory(final_state=state,
                          state_actions=self.state_actions+[(self.final_state, action)])

    def to_program(self):
        from cad import Program
        return Program([k
                        for s,a in self.state_actions
                        for k in a.to_program(s) ])

    def __str__(self):
        return f"Trajectory(final_state={self.final_state}, state_actions={self.state_actions})"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return str(self) == str(other)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(str(self))
