from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from network import RAFTGMA

from torch.utils.data import DataLoader
from utils import flow_viz
import evaluate
import datasets

import logging
import hashlib
from config.config_loader import load_json_config, cpy_args_to_config
import json
# import learning_rate as LR
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    print("NOOOOO GRADSCALE!!EXITING.....")
    exit(0)
    # dummy GradScaler for PyTorch < 1.6 removed from the original raft code.

# exclude extremly large displacements
MAX_FLOW = 400


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(flow_preds)    
    flow_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    valid = (valid >= 0.5) & ((flow_gt**2).sum(dim=1).sqrt() < MAX_FLOW)
   
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        "flow_loss": flow_loss,
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fetch_optimizer(config, phase, model, local_step):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr= config["train"]["lr"][phase], weight_decay= config["train"]["wdecay"][phase], eps= config["epsilon"])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["train"]["lr"][phase], total_steps=config["train"]["num_steps"][phase]+100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    for i in range(local_step + 1):
        scheduler.step()

    return optimizer, scheduler
    

class StatsLogger:
    def __init__(self, name, current_steps, phase):
        self.total_steps = current_steps
        self.phase = phase
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir= os.path.join("checkpoints", name))
        self.metrics_file = os.path.join("checkpoints", name, "lrs.csv")
        self.time = datetime.now()
        self.logger = logging.getLogger("gma.stats") # this is a logger of type "gma". It can be more strict.
        
        # self.train_epe_list = []
        # self.train_steps_list = []
        # self.val_steps_list = []
        # self.val_results_dict = {}

        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, "w") as file:
                    file.write("step,lr\n") 

    def set_phase(self, phase, dataset):
        self.phase = phase
        self.dataset_being_trained = dataset

    def _print_training_status(self, lr):
        # metrics_data = [np.mean(self.running_loss[k]) for k in sorted(self.running_loss.keys())]
        metrics_data = [self.running_loss[k]/config["print_freq"]for k in sorted(self.running_loss.keys())]
        
        # current time
        now = datetime.now()
        time_diff = now - self.time
        self.time = now

        # # Compute time left
        # time_left_sec = (self.args.num_steps - (self.total_steps+1)) * metrics_data[-1]
        # time_left_sec = time_left_sec.astype(np.int)
        # time_left_hms = "{:02d}h{:02d}m{:02d}s".format(time_left_sec // 3600, time_left_sec % 3600 // 60, time_left_sec % 3600 % 60)
        # time_left_hms = f"{time_left_hms:>12}"

        training_str = "[number of steps: {0:6d}, lr: {1:2.7f}, dataset: {2}, phase: {3}, duration: {4:4.2f}, time:{5}] ".format(self.total_steps+1, lr, self.dataset_being_trained, self.phase, time_diff.total_seconds(), now)
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        # print (training_str + metrics_str)
        self.logger.debug(", ".join(k for k in sorted(self.running_loss.keys())))
        self.logger.info("%s %s", training_str, metrics_str)
        
        # # logging running loss to total loss
        # self.train_epe_list.append(np.mean(self.running_loss_dict['epe']))
        # self.train_steps_list.append(self.total_steps)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/config["print_freq"], self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics, lr):
        self.total_steps += 1 # assume local step starts from -1, as it actually does.

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        with open(self.metrics_file, "a") as file:
            file.write("{:6d},{:10.7f}\n".format(self.total_steps, lr)) 

        if self.total_steps % config["print_freq"] == config["print_freq"]-1:
            self._print_training_status(lr)
            self.running_loss = {}

    def write_dict(self, results):

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()




def save_model_and_checkpoint(model, config, steps, phase, saving_policy = "limited"):
    if saving_policy == "unlimited":
        checkpoint_values_path = 'checkpoints/%s/_%s_phase%d_%d.pth' % (config["name"], config["train"]["dataset"][phase], phase, steps)
        torch.save(model.state_dict(), checkpoint_values_path)
    elif saving_policy == "limited":
        checkpoint_values_path = 'checkpoints/%s/%s_phase%d_%d.pth' % (config["name"], config["train"]["dataset"][phase], phase, steps)
        torch.save(model.state_dict(), checkpoint_values_path)
        checkpoint_txt_path = 'checkpoints/%s/checkpoint.txt' % config["name"]
        create_checkpoint_file(checkpoint_txt_path, phase, steps, checkpoint_values_path, config) 
    else:
        assert ValueError("Wrong saving policy given.")

def create_checkpoint_file(txtfile_path, phase, current_steps, checkpoint_name, config):
    if not os.path.exists(txtfile_path):
            
        with open(txtfile_path, 'w') as file:
            dict = {"phase": phase, "current_steps": current_steps, "newer": checkpoint_name, "older": None}
            json.dump(dict, file)
          
    else:
        with open(txtfile_path) as file:
            checkpoint_config = json.load(file)

        with open(txtfile_path, "w") as file:

            if checkpoint_config["newer"] == None:
                dict = {"phase": phase, "current_steps": current_steps, "newer": checkpoint_name, "older": None}
                json.dump(dict, file)
            elif (checkpoint_config["newer"] != None) and (checkpoint_config["older"] == None):
                dict = {"phase": phase, "current_steps": current_steps, "newer": checkpoint_name, "older": checkpoint_config["newer"]}
                json.dump(dict, file)
            else:
                dict = {"phase": phase, "current_steps": current_steps, "newer": checkpoint_name, "older": checkpoint_config["newer"]}
                json.dump(dict, file)
                # remove the older file:
                name = config["name"]
                older_file_path = checkpoint_config["older"]
                file_path_to_be_removed = older_file_path
                if os.path.exists(file_path_to_be_removed):
                    os.remove(file_path_to_be_removed)
                else:
                    logger = logging.getLogger("gma.saving")
                    logger.error("Checkpoint file did not exist. old checkpoint.txt: %s, new checkpoint.txt: %s", str(checkpoint_config), str(dict))
        
def fetch_model(config, phase):
    model = nn.DataParallel(RAFTGMA(config), device_ids=config["gpus"])

    print(f"Parameter Count: {count_parameters(model)}")            

    model.cuda()
    model.train()

    # if config["train"]["dataset"][phase] != 'chairs':
    #     model.module.freeze_bn()

    # if config["train"]["dataset"][phase] = 'sintel':
    #     model.module.freeze_net()

    return model

def fetch_data(config, phase):
    data_loader = datasets.fetch_dataloader(config, phase)
    while True:
        for data_blob in data_loader:
            yield [x.cuda() for x in data_blob]

def passed_steps(config, phase):
    steps = 0 
    if phase != 0:
        steps = sum(config["train"]["num_steps"][:phase])
        print("phase: %d, step: %d" %(phase, steps))
   
    return steps

def training_step(config, model, data, optimizer, phase, scaler):

    iterations = config["train"]["iters"]
    image1, image2, flow, valid = data

    if config["add_noise"]:
        stdv = np.random.uniform(0.0, 5.0)
        image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
        image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

    flow_predictions = model(image1, image2, iters=iterations)  
    # print(hashlib.sha256((repr(flow_predictions)).encode('utf-8')).hexdigest())          

    loss, metrics = sequence_loss(flow_predictions, flow, valid, config["train"]["gamma"][phase])
    if torch.isnan(loss):
        logger = logging.getLogger("gma.saving")
        logger.error("nan loss during training. Exiting...")
        exit(0)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)               
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip"])
    scaler.step(optimizer)

    scaler.update()
    return metrics


def train_single_phase(model, data_generator, optimizer, scheduler, init_local_step, num_steps, stats_logger, training_step_fn, save_fn, phase):
    for local_step in range(init_local_step, num_steps - 1):
        optimizer.zero_grad()
        data = next(data_generator)
        metrics = training_step_fn(model, data, optimizer, phase)

        scheduler.step()

        stats_logger.push(metrics, scheduler.get_last_lr()[0])

        if (local_step + 1) % config["val_freq"] == config["val_freq"] - 1: #save checkpoint now
            save_fn(model, local_step + 2, "limited")
            results = {}
    
            if config["train"]["validation"][phase] == 'chairs':
                results.update(evaluate.validate_chairs(model.module))
            elif config["train"]["validation"][phase] == 'sintel':
                results.update(evaluate.validate_sintel(model.module, iters=32))
            elif config["train"]["validation"][phase] == 'sintel_split_sep':
                results.update(evaluate.validate_sintel_split_sep(model.module, iters=32))
            elif config["train"]["validation"][phase] == 'kitti':
                results.update(evaluate.validate_kitti(model.module))

            stats_logger.write_dict(results)
            model.train()

            # if config["train"]["dataset"][phase] != 'chairs':
            #     model.module.freeze_bn()

            # if config["train"]["dataset"][phase] = 'sintel':
            #     model.module.freeze_net()


def train_phases(init_phase, num_phases, local_step_of_init_phase, stats_logger, state_dict, datasets, num_steps, model_fn, data_fn, optimizer_scheduler_fn, training_step_fn, save_fn):
    local_step = local_step_of_init_phase
    for phase in range(init_phase, num_phases):
        stats_logger.set_phase(phase, datasets[phase])
        model = model_fn(phase)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        
        data = data_fn(phase)
        optimizer, scheduler = optimizer_scheduler_fn(phase, model, local_step)
        train_single_phase(model, data, optimizer, scheduler, local_step, num_steps[phase], stats_logger, training_step_fn, lambda model, step, policy: save_fn(model, step, phase, policy), phase)

        save_fn(model, num_steps[phase], phase, "unlimited")

        local_step = -1
        state_dict = model.state_dict()


def train(config):

    logger = logging.getLogger("gma.train")
    logger.info(config)
    if config["train"]["restore_ckpt"] is None:
        possible_checkpoint_file = os.path.join( "checkpoints", config["name"], "checkpoint.txt")
        if (os.path.exists(possible_checkpoint_file)):
            file = open(possible_checkpoint_file)
            checkpoint_configs = json.load(file)
            config["current_phase"] = checkpoint_configs["phase"]
            config["train"]["restore_ckpt"] = checkpoint_configs["newer"]
            config["current_steps"] = checkpoint_configs["current_steps"] - 1 # local step is the index of the steps.         
  
    if config["train"]["restore_ckpt"] is not None:
        state_dict = torch.load(config["train"]["restore_ckpt"])
        print("Loading checkpoint from %s....." %config["train"]["restore_ckpt"])
    else:
        state_dict = None

    total_phase_len = len(config["train"]["num_steps"])
    init_phase = config["current_phase"]
    local_steps = config["current_steps"] # local step is the last step (current step) in the current phase.

    passed_train_steps = passed_steps(config, init_phase)   
    stats_logger = StatsLogger(config["name"], local_steps + passed_train_steps, init_phase)
    scaler = GradScaler(enabled=config["mixed_precision"])
    train_phases(init_phase, total_phase_len, local_steps, stats_logger, state_dict, config["train"]["dataset"], config["train"]["num_steps"],
    lambda phase: fetch_model(config, phase),
    lambda phase: fetch_data(config, phase),
    lambda phase, model, local_step: fetch_optimizer(config, phase, model, local_step),
    lambda model, data, optimizer, phase: training_step(config, model, data, optimizer, phase, scaler),
    lambda model, step, phase, policy: save_model_and_checkpoint(model, config, step, phase, policy))

    stats_logger.close()
    
    print("--------------Reached the end of training. Exiting....--------------")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help="name your experiment")
    parser.add_argument('--dataset', help="which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')#

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])#
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')#

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')#
    parser.add_argument('--add_noise', action='store_true')# 
    parser.add_argument('--val_freq', type=int, default=10000,
                        help='validation frequency')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='printing frequency')

    parser.add_argument('--upsample-learn', action='store_true', default=False,
                        help='If True, use learned upsampling, otherwise, use bilinear upsampling.')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')                    
    
    parser.add_argument('--current_phase', type=int, default=0)
    parser.add_argument('--current_steps', type=int, default= -1)
    parser.add_argument('--config', help= 'path to the configuration file')

    args = parser.parse_args()
    if args.config is not None:
        config = load_json_config(args.config) 
    else:
        config = cpy_args_to_config(args)
    
   
    torch.manual_seed(1234)
    np.random.seed(1234)
    #uncomment below for determinism
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.isdir(f'checkpoints/{config["name"]}'):
        os.mkdir(f'checkpoints/{config["name"]}')

    filehandler = logging.FileHandler(f"checkpoints/{config['name']}/log.txt")
    # In the file, write Info or the other things with higer lever than info: error, warning and stuff.
    filehandler.setLevel(logging.INFO)

    streamhandler = logging.StreamHandler()
    streamhandler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(name)s:%(levelname)s:%(message)s")
    filehandler.setFormatter(formatter)
    streamhandler.setFormatter(formatter)

    logger = logging.getLogger("gma")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    logger.info("starting to train")

    print(config)
    train(config)
