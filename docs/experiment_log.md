# Experiment Log

## Goal

Build a compact autonomous-driving VLA baseline that predicts driving actions from visual observations, language instructions, and vehicle state.

## Current Setup

Inputs:

- Front camera image: synthetic RGB tensor
- Language instruction: one of several driving intents, such as `turn left`, `keep lane`, or `stop`
- Vehicle state: compact ego-state vector

Output:

- Continuous action vector: `[steering, throttle, brake]`

## Model

The baseline uses a three-branch architecture:

- CNN image encoder
- Embedding-based instruction encoder
- MLP vehicle-state encoder

The three feature vectors are concatenated and passed into an MLP action head.

## Training Objective

The project uses imitation learning with mean squared error between predicted and target action vectors.

```text
loss = MSE(predicted_action, target_action)
```

## Metrics

Evaluation reports:

- Overall MSE
- Steering MAE
- Throttle MAE
- Brake MAE

## Why This Matters For VLA

The project demonstrates the core VLA pattern:

```text
vision + language + state -> unified representation -> action
```

For a production or research system, this scaffold can be extended with real driving datasets, pretrained vision-language models, trajectory heads, and simulator-based closed-loop evaluation.
