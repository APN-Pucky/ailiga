from ailiga.APNPucky.DQNFighter.v0.DQNFighter_v0 import DQNFighter_v0
from ailiga.APNPucky.DQNFighter.v1.DQNFighter_v1 import DQNFighter_v1
from ailiga.APNPucky.DQNFighter.v2.DQNFighter_v2 import DQNFighter_v2
from ailiga.APNPucky.RandomFigher.v0.RandomFighter_v0 import RandomFighter_v0


def get_all_fighters():
    """Get all fighters."""
    return [
        RandomFighter_v0,
        DQNFighter_v0,
        DQNFighter_v1,
        DQNFighter_v2,
    ]


def get_fighter_by_name(name):
    """Get a fighter by name."""
    for fighter in get_all_fighters():
        if fighter.get_name() == name:
            return fighter
