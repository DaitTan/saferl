#!/usr/bin/env python3
from lnn import (
    Predicate, Variable,
    Implies, ForAll,
    Model, World, Not,
    Fact
)

class LNNShielding:
    def __init__(self):
        self.model = Model()

        action, state = map(Variable, ["action", "state"])

        self.take_action = Predicate("take_action", world=World.CLOSED)
        self.is_one_square_away = Predicate("one_square_away")

        self.if_enemy_close_dont_go = ForAll(
            state, action,
            Implies(self.is_one_square_away(state),
            Not(self.take_action(action))))

        self.model.add_knowledge(self.if_enemy_close_dont_go, world=World.AXIOM)

    def is_action_safe(self, num_close_ghosts: int) -> bool:
        self.model.flush()
        self.model.add_data({self.take_action: {"action": Fact.TRUE}})
        for i in range(num_close_ghosts):
            self.model.add_data({self.is_one_square_away: {str(i): Fact.TRUE}})

        self.model.infer()
        # If the model has a contradiction then the action isn't safe
        return not self.model.has_contradiction()