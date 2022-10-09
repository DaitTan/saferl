#!/usr/bin/env python3
from lnn import Predicate, Variable, Join, And, Exists, Implies, ForAll, Model, Fact, Or

def func():
    """The 'American' theorem proving example with  inner joins"""

    s, a = map(Variable, ["s", "a"])
    model = Model()  # Instantiate a model.

    # Define and add predicates to the model.
    # Predicate for what action the RL model selected 
    move_up = Predicate("move_up")
    move_down = Predicate("move_down")
    move_left = Predicate("move_left")
    move_right = Predicate("move_right")

    # Predicates to represent the state. Is this possible like this?
    # Can we make the closed world assumption that we know the safety
    # Of each direction? If we did then can't we just avoid it w/o LNN
    up_unsafe = Predicate("up_unsafe")
    down_unsafe = Predicate("down_unsafe")
    left_unsafe = Predicate("left_unsafe")
    right_unsafe = Predicate("right_unsafe")

    # Unsafe don't move there
    is_up_unsafe = And(up_unsafe(s), move_up(a))
    is_down_unsafe = And(down_unsafe(s), move_down(a))
    is_left_unsafe = And(left_unsafe(s), move_left(a))
    is_right_unsafe = And(right_unsafe(s), move_right(a))

    is_action_unsafe = Or(is_up_unsafe(s, a), is_down_unsafe(s, a), is_left_unsafe(s, a), is_right_unsafe(s, a))

    true_fact = {"0": Fact.FALSE, "1": Fact.TRUE}
    false_fact = {"0": Fact.TRUE, "1": Fact.FALSE}

    model.add_knowledge(is_action_unsafe)

    # Up is our move and is unsafe
    model.add_data({
        move_up: true_fact,
        move_down: false_fact,
        move_left: false_fact,
        move_right: false_fact,

        up_unsafe: true_fact,
        down_unsafe: false_fact,
        left_unsafe: true_fact,
        right_unsafe: false_fact,
    })
    model.infer()
    model.print()

    # Up is our move and is unsafe
    #move_up.add_data(true_fact)
    #move_down.add_data(false_fact)
    #move_left.add_data(false_fact)
    #move_right.add_data(false_fact)

    #up_unsafe.add_data(true_fact)
    #down_unsafe.add_data(false_fact)
    #left_unsafe.add_data(true_fact)
    #right_unsafe.add_data(false_fact)

    #is_action_unsafe.upward()

    #print(is_action_unsafe.state())
    #print(move_up.state())
    #print(move_down.state())
    #print(move_left.state())
    #print(move_right.state())


if __name__ == "__main__":
    func()
