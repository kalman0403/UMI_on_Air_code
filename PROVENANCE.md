# Provenance and local modifications

- **Upstream**: this tree is a snapshot of **UMI-on-Air** (https://github.com/LeCAR-Lab/UMI-on-Air).
  The bundled UMI components are MIT-licensed — see
  `am_mujoco_ws/universal_manipulation_interface/LICENSE` (Copyright (c) 2023 Columbia
  Artificial Intelligence and Robotics Lab). Third-party licences
  (e.g. `am_mujoco_ws/mujoco_menagerie/universal_robots_ur10e/LICENSE`) are kept in place.
- **Not included**: model checkpoints, demonstration datasets and evaluation outputs.
  The authors publish checkpoints separately on Hugging Face; download `checkpoints.tar.gz`
  and unpack it so that `checkpoints/umi_{cabinet,peg,valve}/` sits next to `am_mujoco_ws/`.
- **Local modifications** (each marked with a `[PATCH bug#N]` comment in place):
  1. `am_mujoco_ws/policy_learning/imitate_episodes.py` — cap the number of MPC restarts per
     rollout (`MAX_MPC_RESTARTS_PER_ROLLOUT = 10`). Without the cap, a degenerate guidance
     setting (`--scale > 0 --guided_steps 0`) restarted the same episode index forever;
     after the cap the episode is recorded as a failure and the run terminates normally.
  2. `am_mujoco_ws/policy_learning/imitate_tracking_utils.py` — tolerate the missing
     `ref_vs_mpc_*` / `mpc_vs_actual_*` comparison keys (`.get(key, 0.0)`), which previously
     raised `KeyError` at finalisation whenever `--log_diffusion` was combined with guidance,
     preventing `metrics.json` / `experiment_summary.json` from being written.
- Evaluation reads the configuration embedded in the checkpoint, so `imitate_episodes.py`
  runs as-is (behaviour documented in the evaluation notes, which are not part of this repo).
