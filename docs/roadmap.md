# Roadmap

## Phase 1: Repository Foundation

- Keep the config, model, policy, and data interfaces small.
- Add smoke tests for every public interface.
- Document the intended benchmark and dataset.

## Phase 2: Dataset Adapter

- Add adapters for one real driving dataset or simulator log.
- Normalize samples into observation, instruction, vehicle state, action, and metadata fields.
- Add data validation and dataset statistics.

## Phase 3: Model Baseline

- Add a vision encoder.
- Add a language encoder.
- Fuse multimodal embeddings.
- Predict steering, throttle, brake, or trajectory outputs through an action head.

## Phase 4: Training and Evaluation

- Add checkpointing.
- Add offline imitation learning metrics.
- Add rollout-based evaluation in CARLA or another simulator.
- Add reproducible experiment reports.

## Phase 5: Release

- Add a model card.
- Add dataset documentation.
- Add reproducibility instructions.
- Publish a first benchmark result.
