#!/bin/bash

poetry run ruff check sudoku_rl test
poetry run mypy -p sudoku_rl -p test
