[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build status](https://github.com/ankandrew/DeepSudoku/actions/workflows/ci/badge.svg)](https://github.com/ankandrew/DeepSudoku/actions/workflows/ci/badge.svg)


![Deep Sudoku Image](assets/Sudoku_Github.png "Title")


### Intro

Using neural nets to learn how to play Sudoku through Reinforcement Learning.

### Requirements

Python 3.7+ is required.

### Generate


```python
import numpy as np

from sudoku_rl import sudoku_generator

solved_sudoku, unsolved_sudoku = sudoku_generator.generate_9x9_sudoku()

print(solved_sudoku)
np.array(
    [
        [1, 5, 8, 9, 6, 7, 4, 3, 2],
        [2, 4, 3, 5, 8, 1, 6, 7, 9],
        [9, 6, 7, 4, 3, 2, 8, 1, 5],
        [5, 8, 1, 6, 7, 9, 3, 2, 4],
        [4, 3, 2, 8, 1, 5, 7, 9, 6],
        [6, 7, 9, 3, 2, 4, 1, 5, 8],
        [8, 1, 5, 7, 9, 6, 2, 4, 3],
        [3, 2, 4, 1, 5, 8, 9, 6, 7],
        [7, 9, 6, 2, 4, 3, 5, 8, 1],
    ],
    dtype=np.int8,
)

print(unsolved_sudoku)
np.array(
    [
        [0, 0, 0, 0, 0, 6, 0, 3, 0],
        [0, 0, 0, 0, 3, 0, 0, 0, 0],
        [8, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 4, 0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 0, 5, 0],
        [0, 1, 0, 0, 5, 0, 0, 0, 3],
        [0, 0, 0, 0, 7, 0, 0, 0, 5],
        [0, 0, 0, 0, 9, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 8, 3, 0, 1],
    ],
    dtype=np.int8,
)
```

### Validate

The [Validator](sudoku_rl/sudoku_validator.py) module is responsible for validating a sudoku.
For example, we can validate the following Sudoku:

```python
from sudoku_rl import sudoku_validator
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

print(sudoku_validator.is_sudoku_valid(sudoku))
```

### Train

To train run `train.py`. Usage:

<!---
TODO: Complete this
-->
