#!/bin/bash

poetry run pylint sudoku_rl test
poetry run mypy -p sudoku_rl -p test
