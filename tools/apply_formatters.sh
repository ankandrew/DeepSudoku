#!/bin/bash

poetry run isort sudoku_rl test
poetry run black sudoku_rl test
