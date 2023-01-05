import re

from ailiga.APNPucky.DQNFighter.v0.DQNFighter_v0 import DQNFighter_v0
from ailiga.APNPucky.DQNFighter.v1.DQNFighter_v1 import DQNFighter_v1
from ailiga.APNPucky.DQNFighter.v2.DQNFighter_v2 import DQNFighter_v2
from ailiga.APNPucky.IndexFighter.v0.IndexFighter_v0 import IndexFighter_v0
from ailiga.APNPucky.RandomFigher.v0.RandomFighter_v0 import RandomFighter_v0
from ailiga.APNPucky.RandomIndexFighter.v0.RandomIndexFighter_v0 import (
    RandomIndexFighter_v0,
)


def get_all_fighters():
    """Get all fighters."""
    return [
        RandomIndexFighter_v0,
        IndexFighter_v0,
        RandomFighter_v0,
        DQNFighter_v0,
        DQNFighter_v1,
        DQNFighter_v2,
    ]


def search_unique_list(crit, a_fighter):
    cur = None
    for fighter in a_fighter:
        if crit(fighter.get_name()):
            if cur is None:
                cur = fighter
            else:
                cur = None
                break
    return cur


def get_fighter_by_name(name):
    """Get a fighter by name."""
    # check if fighter is unique in the end of the fighter list of strings
    fighters = get_all_fighters()
    return (
        search_unique_list(lambda x: re.match(name, x), fighters)
        or search_unique_list(lambda x: name in x, fighters)
        or search_unique_list(lambda x: x.endswith(name), fighters)
        or search_unique_list(lambda x: x == name, fighters)
    )


def get_fighters_from_list(a_fighter):
    fghts = []
    if not a_fighter:
        fghts = get_all_fighters()
    else:
        fghts = [get_fighter_by_name(agent) for agent in a_fighter]
    return fghts
