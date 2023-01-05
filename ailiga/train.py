import argparse

from ailiga import env
from ailiga.all_fighters import (
    get_all_fighters,
    get_fighter_by_name,
    get_fighters_from_list,
)
from ailiga.trained_fighter import TrainedFighter


def train(a_fighter=None, a_env=None, a_force=False):
    """Train a fighter or all fighters."""
    if not a_env:
        envs = env.get_envs().keys()
    else:
        envs = a_env
    for e in envs:
        fghts = get_fighters_from_list(a_fighter)
        if not a_force:
            # get all fighters that are valid for the given env
            fghts = [
                a for a in fghts if a.valid_env(e) if issubclass(a, TrainedFighter)
            ]
        if fghts:
            for fg in fghts:
                f = fg(env.get_env(e))
                f.train()
        else:
            print("No fighters found for env", e)


def main():
    parser = argparse.ArgumentParser(description="Train fighters.")
    parser.add_argument(
        "--fighter",
        type=str,
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="force non checked constellations",
        default=False,
    )
    parser.add_argument("--env", type=str, nargs="+", default=[])
    args = parser.parse_args()
    train(args.fighter, args.env, args.force)


if __name__ == "__main__":
    main()
