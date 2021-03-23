[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://travis-ci.com/ankandrew/DeepSudoku.svg?branch=main)](https://travis-ci.com/ankandrew/DeepSudoku)


![Deep Sudoku Image](img/Sudoku_Github.png "Title")


### Intro

Using neural nets to learn how to play Sudoku through Supervised Learning.

*Implemented using PyTorch framework*

### Requirements

Python 3.7+ is required among with the packages found in [requirements.txt](requirements.txt)

### Generate

A sudoku is firt generated and then is masked with 0's with certain probability. The latter is used as
**feature**/**input** to the model while the former one is used as the **ground truth**.

#### Single

```python
from deep_sudoku.data.generator import Generator

gen = Generator()
x, y = gen()
```

###### X sample

```python
array(
    [
        [4, 1, 3, 0, 0, 2, 0, 0, 0],
        [0, 5, 2, 0, 6, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 3, 0, 0, 2],
        [0, 2, 7, 6, 0, 0, 0, 0, 0],
        [0, 8, 0, 0, 0, 9, 0, 2, 0],
        [0, 0, 9, 0, 0, 7, 0, 8, 4],
        [0, 9, 5, 0, 7, 6, 0, 0, 0],
        [2, 0, 0, 8, 0, 1, 3, 9, 0],
        [0, 0, 0, 0, 0, 0, 0, 7, 0]
    ]
)
```

###### Y sample

```python
array(
    [
        [4, 1, 3, 9, 5, 2, 7, 6, 8],
        [9, 5, 2, 7, 6, 8, 4, 1, 3],
        [7, 6, 8, 4, 1, 3, 9, 5, 2],
        [5, 2, 7, 6, 8, 4, 1, 3, 9],
        [6, 8, 4, 1, 3, 9, 5, 2, 7],
        [1, 3, 9, 5, 2, 7, 6, 8, 4],
        [3, 9, 5, 2, 7, 6, 8, 4, 1],
        [2, 7, 6, 8, 4, 1, 3, 9, 5],
        [8, 4, 1, 3, 9, 5, 2, 7, 6]
    ]
)
```

#### Multiple

For example, to generate a dataset of 10 000 Sudokus:

```python
from deep_sudoku.data.generator import Generator

gen = Generator()
x, y = gen.generate_dataset(n=10_000)
```

### Validate

The [Validator](deep_sudoku/data/validator.py) class is responsible for validating a sudoku. For example If we want to
validate the following Sudoku

```python
from deep_sudoku.data.validator import Validator
import numpy as np

sudoku = np.array(
    [
        [6, 2, 4, 8, 5, 3, 1, 9, 7],
        [8, 5, 3, 1, 9, 7, 6, 2, 4],
        [1, 9, 7, 6, 2, 4, 8, 5, 3],
        [9, 7, 6, 2, 4, 8, 5, 3, 1],
        [2, 4, 8, 5, 3, 1, 9, 7, 6],
        [5, 3, 1, 9, 7, 6, 2, 4, 8],
        [4, 8, 5, 3, 1, 9, 7, 6, 2],
        [3, 1, 9, 7, 6, 2, 4, 8, 5],
        [7, 6, 2, 4, 8, 5, 3, 1, 9]
    ], dtype=np.int8)

val = Validator()
val(sudoku)
```

### Train

To train run `train.py`. Hyper-parameters to tweak can be found in the same file.


