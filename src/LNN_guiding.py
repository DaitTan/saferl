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

def text_world():
    state = Variable("state")

    model = Model()

    visited_all_connected_rooms = Predicate("visited_all_conntected_rooms")
    no_coin_in_east_room = Predicate("no_coin_in_east_room", world=World.CLOSED)
    no_coin_in_west_room = Predicate("no_coin_in_west_room", world=World.CLOSED)
    no_coin_in_north_room = Predicate("no_coin_in_north_room", world=World.CLOSED)
    no_coin_in_south_room = Predicate("no_coin_in_south_room", world=World.CLOSED)
    go_east = Predicate("go_east")
    go_west = Predicate("go_west")
    go_north = Predicate("go_north")
    go_south = Predicate("go_south")
    found_east_room = Predicate("found_east_room", world=World.CLOSED)
    found_west_room = Predicate("found_west_room", world=World.CLOSED)
    found_north_room = Predicate("found_north_room", world=World.CLOSED)
    found_south_room = Predicate("found_south_room", world=World.CLOSED)
    found_coin_in_room = Predicate("found_coin_in_room")
    take_coin = Predicate("take_coin")

    should_not_go_east = Implies(
        And(
            Not(visited_all_connected_rooms(state)),
            no_coin_in_east_room(state)
        ), Not(go_east)
    )
    should_not_go_west = Implies(
        And(
            Not(visited_all_connected_rooms(state)),
            no_coin_in_west_room(state)
        ), Not(go_west)
    )
    should_not_go_north = Implies(
        And(
            Not(visited_all_connected_rooms(state)),
            no_coin_in_north_room(state)
        ), Not(go_north)
    )
    should_not_go_south = Implies(
        And(
            Not(visited_all_connected_rooms(state)),
            no_coin_in_south_room(state)
        ), Not(go_south)
    )

    should_go_east = Implies(found_east_room(state), go_east(state))
    should_go_west = Implies(found_west_room(state), go_west(state))
    should_go_north = Implies(found_north_room(state), go_north(state))
    should_go_south = Implies(found_south_room(state), go_south(state))

    should_take_coin = Implies(found_coin_in_room(state), take_coin(state))
    
    model.add_knowledge(
        should_go_east,
        should_go_west,
        should_go_north,
        should_go_south,
        should_not_go_east,
        should_not_go_west,
        should_not_go_north,
        should_not_go_south,
        should_take_coin,
        world=World.AXIOM
    )

    # Set up an example where the north and west room are found
    # And there is a coin in the north room but not south
    model.add_data({
        no_coin_in_north_room: {"state": Fact.FALSE},
        no_coin_in_south_room: {"state": Fact.TRUE},
        visited_all_connected_rooms: {"state": Fact.FALSE},
        found_coin_in_room: {"state": Fact.FALSE},
        found_north_room: {"state": Fact.TRUE},
        found_south_room: {"state": Fact.TRUE},
    })
    model.infer()
    model.print()
    logger.info(f'Go east: {go_east.state()}')
    logger.info(f'Go west: {go_west.state()}')
    logger.info(f'Go north: {go_north.state()}')
    logger.info(f'Go south: {go_south.state()}')
    # logger.info(f'Go east: {go_east.groundings}')
    # logger.info(f'Go west: {go_west.groundings}')
    # logger.info(f'Go north: {go_north.groundings}')
    # logger.info(f'Go south: {go_south.groundings}')

def pacman():
    action, state = map(Variable, ["action", "state"])
    state = Variable("state")

    model = Model()

    take_action = Predicate("take_action", world=World.CLOSED)
    is_one_square_away = Predicate("one_square_away")

    if_enemy_close_dont_go = ForAll(
        state, action, Implies(is_one_square_away(state), Not(take_action(action))))

    model.add_knowledge(if_enemy_close_dont_go, world=World.AXIOM)

    # model.add_data({is_one_square_away: {"blinky": (0.8, 1.0)}})
    # model.add_data({is_one_square_away: {"clyde": (0.2, 0.4)}})
    # model.add_data({is_one_square_away: {"pinky": (0.2, 0.4)}})
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

@hydra.main(config_path="../configs", config_name="lnn_guide", version_base="1.2")
def main(cfg):
    hydra.utils.instantiate(cfg)

if __name__ == "__main__":
    main()