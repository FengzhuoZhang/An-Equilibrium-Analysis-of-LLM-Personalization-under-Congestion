# Supervised Fine-Tuning vs. In-Context Learning: An Equilibrium Analysis of LLM Personalization under Congestion

This repository contains code for studying the interaction between LLM
personalization and compute congestion. It has two main parts:

1. `congestion_game/`: a mean-field congestion game simulator where users choose
   between supervised fine-tuning (SFT) and in-context learning (ICL).
2. `personalization/`: transformer experiments for noisy linear regression with
   partial observations, including ICL pretraining, SFT, and checkpoint
   evaluation.

The accompanying manuscript is included as:

```text
Personalization_and_Compute_Congestion_in_LLM_Services_arxiv.pdf
```

## Repository Layout

```text
.
├── congestion_game/
│   └── congestion_game_simulator.py
├── personalization/
│   ├── train_partial.py
│   ├── sft_partial.py
│   ├── eval_partial.py
│   ├── models.py
│   ├── tasks.py
│   ├── samplers.py
│   ├── curriculum.py
│   ├── base_models.py
│   ├── plot_utils.py
│   ├── schema_partial.py
│   ├── schema_partial_sft.py
│   └── conf/
│       ├── base.yaml
│       ├── linear_regression.yaml
│       ├── linear_regression_partial.yaml
│       ├── linear_regression_partial_sft.yaml
│       └── wandb.yaml
├── requirements.txt
└── Supervised Fine-Tuning vs. In-Context Learning: An Equilibrium Analysis of LLM Personalization under Congestion.pdf
```

## Setup

Create a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the listed dependencies:

```bash
pip install -r requirements.txt
```

The current code also imports a few packages that are not listed in
`requirements.txt`:

```bash
pip install scipy omegaconf munch funcy
```

Python 3.8+ is recommended. The personalization experiments call `.cuda()` in
several places, so a CUDA-capable PyTorch installation and GPU are recommended.

## Congestion Game Simulation

The congestion simulator models a two-type user population choosing between SFT
and ICL. Each option has an error function and a resource cost; aggregate
resource demand determines the congestion level.

Main file:

```text
congestion_game/congestion_game_simulator.py
```

Run the default sweep from the repository root:

```bash
mkdir -p output
python congestion_game/congestion_game_simulator.py
```

The active sweep varies `R_SFT - R_ICL` for several values of `R_ICL` and saves:

```text
output/homo_general_linear_R_sft.pdf
```

Core classes:

| Class | Purpose |
| --- | --- |
| `UserType` | Stores one user type's parameters. |
| `SystemParams` | Stores `R_sft`, `R_icl`, base price `p`, and type share `q`. |
| `ErrorFunction` | Interface for SFT and ICL error functions. |
| `CongestionFunction` | Interface for congestion functions `h(R)`. |
| `GeneralMFECalculator` | Computes thresholds, fixed points, adoption regimes, and `R^*`. |
| `SimplifiedErrorFunction` | Paper-style SFT and ICL errors used by the default script. |
| `LinearCongestion` | Implements `h(R) = R`. |
| `QuadraticCongestion` | Implements `h(R) = R^2`. |
| `ExponentialCongestion` | Implements `h(R) = exp(R)`. |

The default script includes commented alternatives for sweeps over `pi`, `r`,
and `sigma_e`. To switch congestion functions, edit the `__main__` block:

```python
congestion_fn = LinearCongestion()
# congestion_fn = QuadraticCongestion()
# congestion_fn = ExponentialCongestion()
```

Congestion-game run notes:

- The script imports `debug_utils` from an absolute local path and calls
  `setup_debugpy(force=True)`. If that helper is unavailable, comment out the
  `sys.path.append(...)`, `from debug_utils import setup_debugpy`, and
  `setup_debugpy(force=True)` lines.
- The script sets `plt.rcParams['text.usetex'] = True`, which requires a local
  LaTeX installation. Set it to `False` if Matplotlib fails while rendering.

## Personalization Experiments

The `personalization/` folder contains GPT-2-style transformer experiments for
linear regression tasks with partial observations. The main workflow is:

1. Pretrain a model for ICL on synthetic tasks.
2. Fine-tune a model with SFT on a fixed sampled dataset.
3. Evaluate checkpoints in-distribution and out-of-distribution over observed
   dimensions.

Run personalization commands from inside the folder so config paths resolve as
intended:

```bash
cd personalization
```

### ICL Pretraining

```bash
python train_partial.py --config conf/linear_regression_partial.yaml
```

Useful overrides:

```bash
python train_partial.py --config conf/linear_regression_partial.yaml \
  --n_dims 30 \
  --reserved_dims 20 \
  --lr 0.0001
```

The default partial-regression config writes outputs under:

```text
personalization/models/noisy_linear_regression_partial/
```

Each run creates a parameterized run directory containing `config.yaml`,
`state.pt`, saved model checkpoints, and `task.pt`.

### Supervised Fine-Tuning

```bash
python sft_partial.py --config conf/linear_regression_partial_sft.yaml \
  --trained_model_path ./models/noisy_linear_regression_partial/<run_dir>/<run_id> \
  --dataset_size 1024 \
  --epochs 100 \
  --bsize 64 \
  --no_wandb
```

Useful SFT overrides:

| Flag | Description |
| --- | --- |
| `--trained_model_path` | Directory containing the pretrained checkpoint. |
| `--dataset_size` | Number of examples in the fixed SFT dataset. |
| `--epochs` | Number of SFT passes over the dataset. |
| `--bsize` | Mini-batch size. |
| `--keep_every_epochs` | Epoch checkpoint frequency. |
| `--no_wandb` | Disable Weights & Biases logging. |

The SFT script currently loads `model_80000.pt` from `--trained_model_path`.
Make sure that checkpoint exists, or update the filename in `sft_partial.py`.

### Evaluation

Before evaluating, edit the checkpoint path near the top of
`personalization/eval_partial.py`:

```python
path = "./models/noisy_linear_regression_partial/.../<run_id>"
```

Then run:

```bash
python eval_partial.py
```

The script prints in-distribution errors and writes OOD results to:

```text
<checkpoint_path>/eval_results.pkl
```

## Configuration

Training and fine-tuning configs are in `personalization/conf/` and are parsed
with Quinine.

| File | Purpose |
| --- | --- |
| `base.yaml` | Shared model/training defaults and W&B inheritance. |
| `linear_regression.yaml` | Basic linear-regression ICL config. |
| `linear_regression_partial.yaml` | Partial-observation ICL pretraining config. |
| `linear_regression_partial_sft.yaml` | Partial-observation SFT config. |
| `wandb.yaml` | W&B project/entity/logging settings. |

Important config keys:

| Parameter | Description |
| --- | --- |
| `model.n_dims` | Latent input dimension. |
| `model.n_positions` | Maximum context length. |
| `model.n_embd` | Transformer embedding dimension. |
| `model.n_layer` | Number of transformer layers. |
| `model.n_head` | Number of attention heads. |
| `training.task` | Task name, such as `noisy_linear_regression_partial`. |
| `training.batch_size` | Training batch size. |
| `training.learning_rate` | Optimizer learning rate. |
| `training.train_steps` | Number of ICL training steps. |
| `training.dataset_size` | Fixed SFT dataset size. |
| `training.epochs` | Number of SFT epochs. |
| `training.curriculum.points` | Curriculum schedule for context length. |
| `training.curriculum.dims` | Curriculum schedule for active dimensions. |
| `task.reserved_dims` | Number of observed dimensions. |
| `task.reserved_basis_type` | Observation basis, either `eye` or `randn`. |
| `task.noise_std` | Label noise standard deviation. |
| `task.device` | `cuda` or `cpu`. |

To configure Weights & Biases, edit:

```text
personalization/conf/wandb.yaml
```

and set:

```yaml
wandb:
  entity: your-wandb-entity
```

## Current Checkout Notes

This checkout appears to be a partial research snapshot. Before running all
experiments end to end, check the following:

- `personalization/conf/base.yaml` inherits `models/standard.yaml`, but that
  file is not present in the current repository. Restore it or inline the
  missing model keys, especially `model.family`, `model.n_embd`,
  `model.n_layer`, and `model.n_head`.
- Several personalization files import `eval` helpers, but `eval.py` is not
  present in the current repository. Restore the missing helper file or remove
  unused imports where appropriate.
- The personalization code assumes CUDA in multiple places through `.cuda()`
  calls. CPU-only runs may require code edits in addition to setting
  `task.device: "cpu"`.
- The default training configs are computationally heavy. For smoke tests,
  reduce `training.train_steps`, `model.n_positions`, and the curriculum
  endpoints.
