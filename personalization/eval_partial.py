import json
import os
import sys
import math
from munch import Munch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import yaml
from omegaconf import OmegaConf
import copy
import pickle

from models import build_model
from eval import get_run_metrics
from tasks import get_task_sampler, get_task
from samplers import get_data_sampler


def load_config(config_path):
    config = OmegaConf.load(config_path)
    if 'inherit' in config:
        merged = OmegaConf.create({})

        for base_path in config.inherit:
            base = OmegaConf.load(base_path)
            merged = OmegaConf.merge(merged, base)

        del config.inherit
        config = OmegaConf.merge(merged, config)

    return config


path = "./models/noisy_linear_regression_partial/dim_30_positions_1031_points_1031_res_dims_15_res_type_eye_noise_std_0.5_bsize_64_steps_100001_lr_0.0001/<run_id>"
state_path = os.path.join(path, "model.pt")
config_path = os.path.join(path, "config.yaml")
task_path = os.path.join(path, "task.pt")
state = torch.load(state_path)
args = load_config(config_path)
train_steps = state["train_step"]
model = build_model(args.model)
model.load_state_dict(state["model_state_dict"])
model.cuda()
model.eval()

eval_bsize = 12800
n_points = 1025
mini_batch = 320
n_dims = model.n_dims
reserved_dims = args.task.reserved_dims
reserved_basis_type = args.task.reserved_basis_type
device = args.task.device
noise_std = args.task.noise_std
data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)


task_indist = get_task(
    args.training.task,
    n_dims,
    eval_bsize,
    reserved_dims=reserved_dims,
    reserved_basis_type=reserved_basis_type,
    device=device,
    noise_std=noise_std,
    num_tasks=args.training.num_tasks,
    **args.training.task_kwargs,
)
task_indist.load_task(task_path)
task_indist.to_device("cuda")
batch_errors = []
train_errors = []
for i in range(0, eval_bsize, mini_batch):
    end = min(i + mini_batch, eval_bsize)
    num_samples = end - i
    xs = data_sampler.sample_xs(
        n_points,
        num_samples,
    ).cuda()
    task_indist.b_size = num_samples
    ys, xs_projected = task_indist.evaluate(xs)

    loss_func = task_indist.get_metric()
    train_loss_func = task_indist.get_training_metric()
    with torch.no_grad():
        output = model(xs_projected, ys)
        loss = loss_func(output, ys)
        train_loss = train_loss_func(output, ys)
    batch_errors.append(loss)
    train_errors.append(train_loss)
batch_errors =torch.concat(batch_errors,dim=0)
errors_indist = torch.mean(batch_errors,dim=0)
print("in-dist errors: \n", errors_indist.cpu().numpy())

start_dim_index = 0
end_dim_index = n_dims-1

def eval_ood(start_dim_index, end_dim_index, task_indist, eval_bsize, mini_batch, n_points):
    task_ood = copy.deepcopy(task_indist)
    task_ood.reserved_dims = end_dim_index-start_dim_index+1
    task_ood.reserved_basis = torch.zeros(n_dims,n_dims).cuda()
    task_ood.reserved_basis[:,start_dim_index:end_dim_index+1] = task_ood.full_basis[:,start_dim_index:end_dim_index+1]
    task_ood.x_scale = math.sqrt(n_dims / task_ood.reserved_dims)
    task_ood.to_device("cuda")
    batch_errors = []
    train_errors = []
    for i in range(0, eval_bsize, mini_batch):
        end = min(i + mini_batch, eval_bsize)
        num_samples = end - i
        xs = data_sampler.sample_xs(
            n_points,
            num_samples,
        ).cuda()
        task_ood.b_size = num_samples
        ys, xs_projected = task_ood.evaluate(xs)

        loss_func = task_ood.get_metric()
        train_loss_func = task_ood.get_training_metric()
        with torch.no_grad():
            output = model(xs_projected, ys)
            loss = loss_func(output, ys)
            train_loss = train_loss_func(output, ys)
        batch_errors.append(loss)
        train_errors.append(train_loss)
    batch_errors =torch.concat(batch_errors,dim=0)
    errors_ood = torch.mean(batch_errors,dim=0)
    return errors_ood

data_dict = {}
data_dict["ckpt_path"] = path
data_dict["errors"] = []
end_dim_index_list = list(range(reserved_dims-1,n_dims))
for end_dim_index in end_dim_index_list:
    print(f"Evaluating end_dim_index: {end_dim_index}")
    temp_dict = {}
    temp_dict["start_dim_index"] = start_dim_index
    temp_dict["end_dim_index"] = end_dim_index
    errors_ood = eval_ood(start_dim_index, end_dim_index, task_indist, eval_bsize, mini_batch, n_points)
    errors_ood = errors_ood.cpu().numpy()
    temp_dict["errors_ood"] = errors_ood
    data_dict["errors"].append(temp_dict)

with open(os.path.join(path, 'eval_results.pkl'), 'wb') as f:
    pickle.dump(data_dict, f)


print("finished evaluation")
