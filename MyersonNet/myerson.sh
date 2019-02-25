#!/bin/bash

python main.py -agent 3 -item 1 -distr exponential -num_linear 10 -num_max 5 &
python main.py -agent 3 -item 1 -distr uniform -num_linear 10 -num_max 10 &
python main.py -agent 5 -item 1 -distr asymmetric_uniform -num_linear 10 -num_max 10 &
python main.py -agent 3 -item 1 -distr irregular -num_linear 10 -num_max 5 &