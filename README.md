# RAFT
This repository contains the source code for the paper:

[RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf)<br/>
Zachary Teed and Jia Deng<br/>

## Requirements
The following additional packages need to be installed

  ```Shell
  pip install Pillow
  pip install scipy
  pip install opencv-python
  pip install tqdm
  pip install matplotlib
  pip install tensorboard
  ```

To train the network for all training stages, run


```Shell
python train.py --config config/baseline_fullBatched_mixedPre.json
```

If your GPU has less than 23 GBs of memory run:

```Shell
python train.py --config config/baseline_lessBatched_mixedPre.json
```
Note that using both these files activates mixed precision which enables faster training and less memory usage and comparable intermediate results.
In general if you have access to a Titan GPU (Or any other GPU with more than 23 GBs of memory), I highly recommend that you use 

```Shell
baseline_fullBatched_mixedPre.json
```

To reproduce the results on the benchmark, first please check carefully if you are reading all the datasets correctly and completely. Second, make sure that you train the network via:

```Shell
python train.py --config config/baseline_singlePre.json
```

Importantly this config file does not activate mixed precision training, therefore it takes longer to train and it uses more memory, however to produce the benchmarks' results, this setting is highly recommended. 

Suggestion: train intermediate networks (stuff you try while you do research) using 

```Shell
python train.py --config config/baseline_fullBatched_mixedPre.json
```
and for final training to validate the approach on the benchmarks, use single precision version.

# Notes about loggings that are added compared to the original repo

Whenever you want to initiate training using different settings, make a custom config files with the settings you want. Change the attribute "name".
When you start training, a folder under /checkpoints is created with the name you entered in the config file.
Inside the folder there will be a log file. This log file has info about training: how may iterations are done, when was that and what training stage is being trained and other things. This info can be useful to track some issues.

Another important option that is added here is a more effective space-saving checkpointing.
The checkpoints are saved permanently only at the end of each training stage: After training is completed on Chairs, Things, Sintel-mixed and Kitti.
There are some intermediate checkpoints that are saved every VAL_FREQ iterations, but they get over-written by the new ones to save more space.

If you want to change some parameters and train a specific training stage, you can set the attributes "phase" to the phase you want to train (1 for training on Things, 2 for Sintel-mixed and 3 for Kitti), "current_steps" to 0 if you want to train that stage from the begining,  "newer" to the path of the checkpoint in checkpoint.txt inside your experiment forlder. You can set "older" to null.
Note that there is no need to do anything with checkpoint.txt if you just want to train from the begining. In that case, you just train with the proper config file.

Importantly, sometimes (not really often) the loss becomes NaN and if the network continues training it might diverge to a useless state, therefore the program checks for that and exits if the loss becomes NaN. In such cases just start training again with the same config file and without changing checkpoint.txt. The porgram loads the last checkpoint automatically.

# Further suggestions:

Use screen sessions or tmux, because if you lose internet connection for a momnet there will be problems with your remote access and that causes exiting the program. Run the program inside tmux or a screen session (or a similar alternative) and detach it to avoid such cases.

Some times the GPUs just freeze and the program stops running. Because of such cases, check if the program is running from time to time (it does not have to be that often, perhaps once or twice a day). You can check the time of last entry in the log file (or just attach the session or tmux to check the time of last summary). If it is not longer than 15 minutes, it's fine. If not, run it again with the same config file and don't change checkpoint.txt
Usually time of the last entry is much less than 15 minutes, but if the network is evaluating a dataset because number_of_iterations % VAL_FREQ == 0, logging the next entry takes much more time due to evaluation on a whole dataset.
