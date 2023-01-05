# AILiga
[![PyPI version][pypi image]][pypi link] [![PyPI version][pypi versions]][pypi link]  ![downloads](https://img.shields.io/pypi/dm/smpl_io.svg) [![Documentation Status](https://readthedocs.org/projects/ailiga/badge/?version=latest)](https://ailiga.readthedocs.io/en/latest/?badge=latest)

## Goals

* Monthly releases of session/tournament results
* User folders
* Strict versioning for reproducibility (ocne a version is pushed, gitignore it)

## Installation

```sh
git clone THIS_PROJECT_URL
poerty install
poetry shell
```




## Testing and Training

Currently, training/testing fighters works through the fighter tests.
```sh
python tests/test_dqn_fighter.py
```

## Tensorboard

```sh
tensorboard --logdir log/ --load_fast=false
```


## Limitations

Currently, the implementation through `tianshou.BasePolicy` seems to only support DQNPolicy and also not `Discrete()` observation spaces.

## References

### Frameworks

* https://github.com/Farama-Foundation/PettingZoo
* https://github.com/vwxyzjn/cleanrl
* https://github.com/Farama-Foundation/Gymnasium
* https://github.com/deepmind/open_spiel
* https://github.com/datamllab/rlcard
* https://tianshou.readthedocs.io/en/master/

### Books

* http://incompleteideas.net/book/the-book-2nd.html


## Development

We use black through

### package/python structure:

* https://mathspp.com/blog/how-to-create-a-python-package-in-2022
* https://www.brainsorting.com/posts/publish-a-package-on-pypi-using-poetry/

[doc stable]: https://apn-pucky.github.io/ailiga/index.html
[doc test]: https://apn-pucky.github.io/ailiga/test/index.html

[pypi image]: https://badge.fury.io/py/ailiga.svg
[pypi link]: https://pypi.org/project/ailiga/
[pypi versions]: https://img.shields.io/pypi/pyversions/ailiga.svg

[a s image]: https://github.com/APN-Pucky/ailiga/actions/workflows/stable.yml/badge.svg
[a s link]: https://github.com/APN-Pucky/ailiga/actions/workflows/stable.yml
[a t link]: https://github.com/APN-Pucky/ailiga/actions/workflows/test.yml
[a t image]: https://github.com/APN-Pucky/ailiga/actions/workflows/test.yml/badge.svg

[cc s q i]: https://app.codacy.com/project/badge/Grade/38630d0063814027bd4d0ffaa73790a2?branch=stable
[cc s q l]: https://www.codacy.com/gh/APN-Pucky/ailiga/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=APN-Pucky/ailiga&amp;utm_campaign=Badge_Grade?branch=stable
[cc s c i]: https://app.codacy.com/project/badge/Coverage/38630d0063814027bd4d0ffaa73790a2?branch=stable
[cc s c l]: https://www.codacy.com/gh/APN-Pucky/ailiga/dashboard?utm_source=github.com&utm_medium=referral&utm_content=APN-Pucky/ailiga&utm_campaign=Badge_Coverage?branch=stable

[cc q i]: https://app.codacy.com/project/badge/Grade/38630d0063814027bd4d0ffaa73790a2
[cc q l]: https://www.codacy.com/gh/APN-Pucky/ailiga/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=APN-Pucky/ailiga&amp;utm_campaign=Badge_Grade
[cc c i]: https://app.codacy.com/project/badge/Coverage/38630d0063814027bd4d0ffaa73790a2
[cc c l]: https://www.codacy.com/gh/APN-Pucky/ailiga/dashboard?utm_source=github.com&utm_medium=referral&utm_content=APN-Pucky/ailiga&utm_campaign=Badge_Coverage

[c s i]: https://coveralls.io/repos/github/APN-Pucky/ailiga/badge.svg?branch=stable
[c s l]: https://coveralls.io/github/APN-Pucky/ailiga?branch=stable
[c t l]: https://coveralls.io/github/APN-Pucky/ailiga?branch=master
[c t i]: https://coveralls.io/repos/github/APN-Pucky/ailiga/badge.svg?branch=master

[rtd s i]: https://readthedocs.org/projects/ailiga/badge/?version=stable
[rtd s l]: https://ailiga.readthedocs.io/en/stable/?badge=stable
[rtd t i]: https://readthedocs.org/projects/ailiga/badge/?version=latest
[rtd t l]: https://ailiga.readthedocs.io/en/latest/?badge=latest
