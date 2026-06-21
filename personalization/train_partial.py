import os
from random import randint
import uuid

from quinine import QuinineArgumentParser
from argparse import ArgumentParser
from tqdm import tqdm
import torch
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler, get_task
from samplers import get_data_sampler
from curriculum import Curriculum
from schema_partial import schema
from models import build_model
import sys

import wandb

torch.backends.cudnn.benchmark = True


class WSDLRcheduler:
    """
    Warmup Step Decay Learning Rate Scheduler.
    - Warmup phase: linearly increase LR from 0 to target LR over warmup_steps
    - Step decay phase: reduce LR by decay_factor at specified decay_steps milestones
    """
    def __init__(self, optimizer, base_lr, warmup_steps, total_steps, decay_steps, decay_factor=0.1):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = 0
        self.total_steps = total_steps
        self.decay_factor = decay_factor
        self.decay_steps = self.total_steps*(1-decay_factor)
        self.current_step = 0
        self.current_lr = base_lr
        self.lr_min_ratio = 0.1

    def step(self, step=None):
        """Update learning rate based on current step"""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        if self.current_step < self.warmup_steps:
            self.current_lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            self.current_lr = self.base_lr
            for decay_step in self.decay_steps:
                if self.current_step >= decay_step:
                    self.current_lr = self.base_lr* (self.lr_min_ratio + (1-self.lr_min_ratio)*(self.current_step-decay_step)/(self.total_steps-decay_step))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr

    def get_lr(self):
        """Get current learning rate"""
        return self.current_lr


def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    scheduler = None
    if args.training.use_wsd_scheduler:
        if args.training.warmup_steps is None:
            raise ValueError("warmup_steps must be specified when use_wsd_scheduler is True")
        if args.training.decay_steps is None:
            raise ValueError("decay_steps must be specified when use_wsd_scheduler is True")
        scheduler = WSDLRcheduler(
            optimizer=optimizer,
            base_lr=args.training.learning_rate,
            warmup_steps=0,
            decay_steps=0,
            total_steps=args.training.train_steps,
            decay_factor=args.training.decay_factor,
        )

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    bsize = args.training.batch_size
    reserved_dims = args.task.reserved_dims
    reserved_basis_type = args.task.reserved_basis_type
    device = args.task.device
    noise_std = args.task.noise_std
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    task = get_task(
        args.training.task,
        n_dims,
        bsize,
        reserved_dims=reserved_dims,
        reserved_basis_type=reserved_basis_type,
        device=device,
        noise_std=noise_std,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    for i in pbar:
        if scheduler is not None:
            scheduler.step()

        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        ).cuda()

        ys, xs_projected = task.evaluate(xs, **task_sampler_args)

        loss_func = task.get_training_metric()

        loss, output = train_step(model, xs_projected, ys, optimizer, loss_func)

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys.cuda()).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            log_dict = {
                "overall_loss": loss,
                "excess_loss": loss / baseline_loss,
                "pointwise/loss": dict(
                    zip(point_wise_tags, point_wise_loss.cpu().numpy())
                ),
                "n_points": curriculum.n_points,
                "n_dims": curriculum.n_dims_truncated,
            }
            if scheduler is not None:
                log_dict["learning_rate"] = scheduler.get_lr()
            wandb.log(log_dict, step=i)

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            training_state = {
                "model_state_dict": model.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, os.path.join(args.out_dir, f"model.pt"))

    training_state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_step": i,
    }
    torch.save(training_state, state_path)
    task.save_task(os.path.join(args.out_dir, f"task.pt"))


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 5001
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_model(args.model)
    model.cuda()
    model.train()

    train(model, args)


if __name__ == "__main__":
    regular_parser = ArgumentParser()
    regular_parser.add_argument('--n_dims', type=int, default=-1)
    regular_parser.add_argument('--reserved_dims', type=int, default=-1)
    regular_parser.add_argument('--lr', type=float, default=-1)
    args_regular, unknown = regular_parser.parse_known_args()

    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + unknown
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")

    if args_regular.n_dims >0:
        args.model.n_dims = args_regular.n_dims
    if args_regular.reserved_dims >0:
        args.task.reserved_dims = args_regular.reserved_dims
    if args_regular.lr >0:
        args.training.learning_rate = args_regular.lr

    n_dims = args.model.n_dims
    n_positions = args.model.n_positions
    reserved_dims = args.task.reserved_dims
    reserved_basis_type = args.task.reserved_basis_type
    noise_std = args.task.noise_std
    bsize = args.training.batch_size
    start_step = args.training.train_steps
    n_points = args.training.curriculum.points.end

    basic_info = f"dim_{n_dims}_positions_{n_positions}_points_{n_points}_res_dims_{reserved_dims}_res_type_{reserved_basis_type}_noise_std_{noise_std}_bsize_{bsize}_steps_{start_step}_lr_{args.training.learning_rate}"

    args.wandb.name += "_" + f"points_{n_points}_res_dims_{reserved_dims}_lr_{args.training.learning_rate}_bsize_{bsize}"
    args.wandb.project = "in-context-training-partial"
    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, basic_info+ "/" +run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
