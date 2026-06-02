# AutoDrive VLA

A starter project for Vision-Language-Action models in autonomous driving.

AutoDrive VLA is a clean GitHub-ready base for experiments that connect driving scene observations, natural-language navigation instructions, vehicle state, and driving actions.

## What This Repository Contains

- A small, readable Python package under `src/autodrive_vla`
- Config-driven experiment entry points
- Placeholder data and model interfaces for autonomous-driving VLA workflows
- Smoke tests to keep the scaffold healthy
- A roadmap for turning the starter into a real research project

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
│       └── policy.py
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

## Next Milestones

1. Choose a target domain: lane keeping, turning, lane changing, or route following.
2. Add a real dataset adapter, such as nuScenes, Waymo Open Dataset, BDD100K, or CARLA logs.
3. Replace `ToyDrivingVLAModel` with a vision-language backbone plus a driving action head.
4. Add training metrics, checkpoints, and simulator rollouts.
5. Publish the repository with a clear model card and dataset documentation.

## License

MIT
