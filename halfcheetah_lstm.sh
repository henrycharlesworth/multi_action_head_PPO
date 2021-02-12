#!/bin/sh

python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 128 --num-processes 16 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 4 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --extra-id "_recurrent_lstm" --seed 0 --recurrent-policy --recurrent-type "LSTM"
