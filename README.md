# A repository template

[![Build Status](https://github.com/nabenabe0928/mfhpo-simulator/workflows/Functionality%20test/badge.svg?branch=main)](https://github.com/nabenabe0928/mfhpo-simulator)
[![codecov](https://codecov.io/gh/nabenabe0928/mfhpo-simulator/branch/main/graph/badge.svg?token=ZXWLF1HM2K)](https://codecov.io/gh/nabenabe0928/mfhpo-simulator)

Before copying the repository, please make sure to change the following parts:
3. The URLs to `Build Status` and `codecov` (we need to copy from the `codecov` website) in `README.md`

## Local check

In order to check if the codebase passes Github actions, run the following:

```shell
$ pip install black pytest unittest flake8 pre-commit pytest-cov
$ ./check_github_actions_locally.sh
```
