# Data

Keep large datasets out of git.

Suggested layout:

```text
data/
├── raw/
│   └── original dataset files
├── processed/
│   └── normalized VLA samples
└── README.md
```

A normalized VLA sample should contain:

- `observation`: front camera image, surround-view images, LiDAR BEV, or structured scene state
- `instruction`: natural-language navigation command or driving intent
- `vehicle_state`: speed, acceleration, steering angle, lane state, or ego pose
- `action`: steering, throttle, brake, lane-change decision, or trajectory target
- `metadata`: scene, map, route, weather, timestamp, and sensor identifiers
