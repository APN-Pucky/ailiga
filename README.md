# AILiga

Build of package based on:

* https://mathspp.com/blog/how-to-create-a-python-package-in-2022
* https://www.brainsorting.com/posts/publish-a-package-on-pypi-using-poetry/

## References

* https://github.com/Farama-Foundation/PettingZoo
* https://github.com/vwxyzjn/cleanrl
* https://github.com/Farama-Foundation/Gymnasium
* https://github.com/deepmind/open_spiel
* https://github.com/datamllab/rlcard
* https://tianshou.readthedocs.io/en/master/

## Limitations

Currently, the implementation through `tianshou.BasePolicy` seems to only support DQNPolicy and also not `Discrite()` observation spaces.

## Tensorboard

```sh
tensorboard --logdir log/ --load_fast=false
```

## Installation

```sh
poerty install
poetry shell
```

## Testing and Training

Currently, training/testing fighters works through the fighter tests.
```sh
python tests/test_dqn_fighter.py
```
