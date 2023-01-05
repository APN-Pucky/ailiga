#!/bin/bash
# Season Script

tm=$(date +%Y_%m)
for envi in tictactoe_v3 simple_spread_v2 knights_archers_zombies_v10
do
	mkdir -p docs/source/tournaments/tabs/$envi
	poetry run python ailiga/tournament.py --env $envi --to_file docs/source/tournaments/tabs/$envi/$tm.rst &
done
