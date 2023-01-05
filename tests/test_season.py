from ailiga import all_fighters, env
from ailiga.season import Season, default_season


def test_season():
    s = Season(
        agents=all_fighters.get_all_fighters(),
        envs=env.get_all_envs(),
        name="default",
        n_episodes=10,
    )
    s.fight()
    print(default_season.as_rst())
