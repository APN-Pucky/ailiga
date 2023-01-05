from ailiga.season import Season, default_season


def test_season():
    default_season.fight()
    print(default_season.as_rst())


test_season()
