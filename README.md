# PPO with multi-head actions
This is a modification of the excellent PyTorch implementation of PPO [here](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) to allow for multi-head/ auto-regressive action spaces (also supports LSTMs now whereas that implementation only supported GRUs).

The set-up I am assuming is that the first action head will be a categorical output of the action type. The output of this then feeds into the subsequent action heads. All of the subsequent action heads can optionally take in the observation output as well as any of the outputs from previous action heads. The basic structure is as follows:

![](https://github.com/henrycharlesworth/multi_action_head_PPO/blob/master/imgs/action_head_structure.png?raw=true)


## Initial Test
Initial test is on the [Platform](https://github.com/cycraig/gym-platform) environment. The agent has to choose between 3 action types (run, hop, leap) as well as providing a parameter that tells it how far/high. So essentially we have a categorical output for the action type and a continuous output for the action parameter. The wrapper I use is [here](https://github.com/henrycharlesworth/multi_action_head_PPO/blob/master/envs/platform_wrapper.py). 

Obviously a fairly simple example (but used in a number of papers on dealing with composite action spaces so not entirely trivial). Will be looking at significantly more complex environments in the future.

I trained using the following command:
```
python main.py --env-name "platform" --algo ppo --use-gae --log-interval 1 --num-steps 128 --num-processes 16 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 16 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --extra-id "" --seed 151 --use-action-masks --no-obs-norm --multi-action-head --recurrent-policy --recurrent-type "LSTM"
```

It trains reasonably quickly and this is the trained agent:

![platform result](https://github.com/henrycharlesworth/multi_action_head_PPO/blob/master/imgs/platform_gif.gif?raw=true)
