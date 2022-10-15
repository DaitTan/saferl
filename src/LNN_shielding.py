#!/usr/bin/env python3
import sys
import logging

from pathlib import Path
PARENT_DIR = Path(__file__, '../..').resolve()
sys.path.append(str(PARENT_DIR))

import hydra

from lnn import (
    Predicate, Variable,
    Join, And, Exists,
    Implies, ForAll,
    Model, Fact, Or,
    World, Not, Loss
)

logger = logging.getLogger("myLogger")

def func():
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

def text_world():
    agent = Variable("agent")

    model = Model()

    found_room = Predicate("found_room")
    visited_room = Predicate("visited_room")
    go_to_room = Predicate("go_to_room")

    visited_and_found_implies_dont_go = Implies(
            And(visited_room(agent), found_room(agent)), 
            Not(go_to_room(agent)))

    found_implies_go = Implies(found_room(agent), go_to_room(agent))

    model.add_knowledge(
            visited_and_found_implies_dont_go,
            found_implies_go,
            world=World.AXIOM)
    #query = Exists(agent, go_room(agent))
    #model.set_query(query)

    model.add_data({
        found_room: {"agent": Fact.TRUE},
        visited_room: {"agent": Fact.TRUE},
        go_to_room: {"agent": Fact.TRUE}})
    model.infer()
    model.print()
    logger.info(model.has_contradiction())

    return
    model.flush()
    model.add_data({
        found_room: {"agent": Fact.FALSE},
        visited_room: {"agent": Fact.TRUE},
        go_to_room: {"agent": Fact.TRUE}})
    model.infer()
    model.print()
    logger.info(model.has_contradiction())

    model.flush()
    model.add_data({
        found_room: {"agent": Fact.TRUE},
        visited_room: {"agent": Fact.FALSE},
        go_to_room: {"agent": Fact.TRUE}})
    model.infer()
    model.print()
    logger.info(model.has_contradiction())

    model.flush()
    model.add_data({
        found_room: {"agent": Fact.FALSE},
        visited_room: {"agent": Fact.FALSE},
        go_to_room: {"agent": Fact.TRUE}})
    model.infer()
    model.print()
    logger.info(model.has_contradiction())

def pacman():
    action, state = map(Variable, ["action", "state"])

    model = Model()

    take_action = Predicate("take_action", world=World.CLOSED)
    is_one_square_away = Predicate("one_square_away")

    if_enemy_close_dont_go = ForAll(
        state, action, Implies(is_one_square_away(state), Not(take_action(action))))

    model.add_knowledge(if_enemy_close_dont_go, world=World.AXIOM)

    # model.add_data({is_one_square_away: {"blinky": (0.8, 0.9)}})
    # model.add_data({is_one_square_away: {"clyde": (0.2, 0.4)}})
    # model.add_data({is_one_square_away: {"pinky": (0.2, 0.4)}})
    # model.add_data({is_one_square_away: {"blinky": Fact.FALSE}})
    # model.add_data({is_one_square_away: {"clyde": Fact.FALSE}})
    # model.add_data({is_one_square_away: {"pinky": Fact.FALSE}})
    # model.add_data({is_one_square_away: {"inky": Fact.FALSE}})
    # for g in ["blinky", "clydy", "pinky", "inky"]:
        # model.add_data({is_one_square_away: {g: Fact.TRUE}})
    model.add_data({take_action: {"action": Fact.TRUE}})

    model.infer()
    model.print()
    logger.info(f'Does the model have contradiction: {model.has_contradiction()}')
    logger.info(if_enemy_close_dont_go.state())

    model.flush()
    model.add_knowledge(if_enemy_close_dont_go, world=World.AXIOM)
    model.add_data({is_one_square_away: {"blinky": Fact.TRUE}})
    model.add_data({is_one_square_away: {"clyde": Fact.FALSE}})
    model.add_data({is_one_square_away: {"pinky": Fact.FALSE}})
    model.add_data({is_one_square_away: {"inky": Fact.FALSE}})
    model.add_data({take_action: {"action": Fact.TRUE}})

    model.infer()
    model.print()
    logger.info(f'Does the model have contradiction: {model.has_contradiction()}')
    logger.info(if_enemy_close_dont_go.state())
    # model.train(losses=[Loss.SUPERVISED, Loss.CONTRADICTION])
    # model.train(losses=[Loss.CONTRADICTION])
    # model.print(params=True)

@hydra.main(config_path="../configs", config_name="lnn_shielding", version_base="1.2")
def main(cfg):
    hydra.utils.instantiate(cfg)

if __name__ == "__main__":
    main()