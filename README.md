# AutoDrive VLA

A compact Vision-Language-Action baseline for autonomous-driving policy learning.

AutoDrive VLA explores a core VLA question for driving: how can a model fuse camera observations, natural-language driving instructions, and ego-vehicle state to predict low-level driving actions?

The current baseline predicts:

```text
[steering, throttle, brake]
```

from:

```text
front camera image + language instruction + vehicle state
```

## What This Repository Contains

- A small, readable Python package under `src/autodrive_vla`
- Config-driven experiment entry points
- A PyTorch multimodal VLA model with image, text, and vehicle-state encoders
- A synthetic driving VLA dataset for fast imitation-learning experiments
- Training and evaluation scripts with checkpoint and metric output
- Smoke tests and a roadmap for extending the baseline to real driving datasets

## Project Structure

```text
.
├── configs/
│   └── example.yaml
├── data/
│   └── README.md
├── docs/
│   └── roadmap.md
├── scripts/
│   ├── demo_policy.py
│   ├── eval.py
│   └── train.py
├── src/
│   └── autodrive_vla/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── model.py
│       ├── policy.py
│       ├── torch_dataset.py
│       └── torch_model.py
├── tests/
│   └── test_smoke.py
├── .gitignore
├── LICENSE
├── pyproject.toml
└── README.md
```

## Quick Start

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run a policy demo:

```bash
python scripts/demo_policy.py --config configs/example.yaml
```

Run tests:

```bash
pytest
```

Install PyTorch dependencies for training:

```bash
python -m pip install -e ".[ml,dev]"
```

Train the VLA baseline:

```bash
python scripts/train.py --config configs/example.yaml
```

Evaluate the trained baseline:

```bash
python scripts/eval.py --config configs/example.yaml
```

Training saves:

```text
outputs/checkpoints/driving_vla_baseline.pt
outputs/checkpoints/driving_vla_baseline.metrics.json
```

## Baseline Model

The PyTorch baseline uses three encoders:

```text
camera image -> CNN image encoder
instruction -> embedding-based text encoder
vehicle state -> MLP state encoder
```

The encoded features are concatenated into a unified multimodal representation and passed through an action head:

```text
fused representation -> steering/throttle/brake prediction
```

This is intentionally small enough to run locally, but it mirrors the structure of larger VLA systems used in robotics and autonomous-driving research.

## VLA Task Framing

A typical autonomous-driving VLA sample can be represented as:

```text
observation: front camera image, surround-view images, LiDAR BEV, or scene state
instruction: natural-language navigation command, route hint, or driving intent
vehicle_state: speed, steering angle, acceleration, lane state, or ego pose
action: steering, throttle, brake, lane-change decision, or trajectory waypoint
```

This starter keeps those pieces explicit so the project can evolve toward:

- imitation learning
- offline driving policy learning
- language-guided driving decisions
- end-to-end planning baselines
- simulator evaluation
- dataset and benchmark tooling for driving scenes

## Interview Talking Points

- Built a VLA-style autonomous-driving baseline that fuses visual, language, and vehicle-state inputs.
- Implemented a synthetic imitation-learning dataset with instruction-conditioned driving actions.
- Designed a modular PyTorch model with separate image, text, and state encoders plus a shared action head.
- Added config-driven training and evaluation scripts with MSE and action-wise MAE metrics.
- Structured the project for future real-data adapters such as nuScenes, Waymo Open Dataset, BDD100K, or CARLA logs.

## Next Milestones

1. Choose a target domain: lane keeping, turning, lane changing, or route following.
2. Add a real dataset adapter, such as nuScenes, Waymo Open Dataset, BDD100K, or CARLA logs.
3. Replace the small CNN/text embedding baseline with a pretrained vision-language backbone.
4. Add closed-loop evaluation in CARLA or another simulator.
5. Publish the repository with a clear model card and dataset documentation.

## License

MIT
