# Interview Guide

## 30-Second Project Pitch

AutoDrive VLA is a compact Vision-Language-Action baseline for autonomous driving. It takes a front camera image, a natural-language driving instruction, and vehicle state, then predicts steering, throttle, and brake. I built it to demonstrate the full VLA workflow: data representation, multimodal model architecture, training, evaluation, and a roadmap toward real driving datasets.

## Technical Points To Explain

1. The task formulation is imitation learning for autonomous driving.
2. The model has separate encoders for image, instruction, and vehicle state.
3. The fused representation predicts continuous driving actions.
4. The synthetic dataset lets me validate the full pipeline before adding real datasets.
5. The evaluation reports both overall MSE and action-wise MAE, which helps diagnose steering versus speed-control errors.

## Honest Limitations

- Current data is synthetic, not from a real driving dataset.
- The image encoder is a small CNN rather than a pretrained vision-language backbone.
- Evaluation is open-loop and does not yet include simulator rollouts.

## Next Extensions

- Add CARLA or nuScenes dataset adapters.
- Replace the text embedding with a pretrained language encoder.
- Replace the action head with trajectory prediction.
- Add closed-loop simulation evaluation.
