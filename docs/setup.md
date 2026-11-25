# 环境安装与基础实现

Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Use Python 3.13 and keep both fork repositories as siblings. The `dev` branch of
`mjlab_for_supermimic` is based on public mjlab 1.6.0 and contains the motion-command
extension hooks required by SuperMimic. For the existing local checkout pair:

```bash
cd ~/Github_Repos/WBC/supermimic_ws
git -C mjlab_for_supermimic switch dev
git -C supermimic switch dev
cd supermimic
uv sync --locked --python 3.13
```

After committing and publishing both `dev` branches, a fresh workspace can be
created with:

```bash
mkdir supermimic_ws && cd supermimic_ws
git clone --branch dev git@github.com:H-Jiang-98/mjlab_for_supermimic.git
git clone --branch dev git@github.com:H-Jiang-98/supermimic.git
cd supermimic
uv sync --locked --python 3.13
```

The editable dependency points to `../mjlab_for_supermimic`. PyTorch 2.14.0
(CUDA 13.0) and Warp 1.18.0 are pinned in the lockfile. Create a separate
`supermimic/.venv` here with `uv sync`; an environment copied from another workspace
can retain editable imports pointing to that old checkout. If you rename or move
the workspace, recreate `.venv` at the new location: installed command scripts
and activation scripts contain absolute paths. A normal `uv sync --locked` can
report all packages installed without repairing those paths. Keep a backup of
the old environment until the verification commands below pass.

`patches/mjlab-1.6.0-motion-hooks.patch` is retained for reproducing the hooks on
the original public checkout at `59b6c720698c032388ba8ab9d0f3582f869d34ff`.
The fork's `dev` branch already contains the motion hooks.

The estimator task also uses the fork's `forward_before_rl_signals` option.
For a checkout that predates this option, apply the kinematics patch from `supermimic/`.

```bash
git -C ../mjlab_for_supermimic apply ../supermimic/patches/mjlab-1.6.0-post-step-kinematics.patch
```

### Public mjlab compatibility

The original code depended on private `mjlab_research`. Public mjlab lacks some
of its motion interfaces, so changing the directory name alone is insufficient.
This checkout implements the required extensions in SuperMimic:

- `motion_loader.py`: single files, lists and globs, exclusive clip boundaries,
  root-state arrays, and `body_state()` computed from joint/root kinematics.
- `utils/forward_kinematics.py`: differentiable floating-base hinge/slide
  kinematics, including world-frame body-origin velocities and entity joint order.
- `motion_command.py`: external sampling distributions used by the default
  return-based sampler, clip-end truncation, and reset updates that preserve the
  sampled frame and do not advance unrelated environments.
- ONNX exports use public mjlab's per-observation metadata, with the actor's
  sensor history length and one sample per goal. The legacy history-length key
  is retained as a matching alias for existing deployment tools.

The mjlab fork adds two overridable methods, `_load_motion()` and
`_sample_time_steps()`. The default public tracking command keeps its original
loader and sampling behavior; the SuperMimic task configurations select the
extended command and add a `motion_end` timeout.

Motion files must contain `joint_pos` and `joint_vel` in entity joint order, plus
either the public mjlab `body_*_w` arrays (root at body index 0) or explicit
`root_pos_w`, `root_quat_w`, `root_lin_vel_w`, and `root_ang_vel_w` arrays. Use
world-frame root velocities and wxyz quaternions. Non-root targets are recomputed
with forward kinematics to agree with the world-model reward. The clip must
already match the control frequency (50 Hz for these tasks); an optional `fps`
field is checked, and mismatches are rejected instead of silently changing speed.

### GenCoord state frames

GenCoord is different from the original supermimic definition, with root angular velocity 
in the root body frame (the pelvis for G1). The packed layout is unchanged:

```text
full_state = [p_w(3), quat_wxyz(4), qj(N), v_w(3), omega_b(3), vj(N)]
rep_state  = [z(1), g_b(3), v_b(3), omega_b(3), qj(N), vj(N)]
prediction = [delta_v_b(3), delta_omega_b(3), delta_vj(N)]
```

The `full_state` observation's `root_ang_vel` term reads `root_link_ang_vel_b`.
Representation and reconstruction pass this angular velocity through directly;
only linear velocity changes frame. Predicted velocity increments are expressed
in the current root frame; integration transports angular velocity to the next
root frame as `omega_b_next = E.T @ (omega_b + delta_omega_b)`, where
`E = Exp(dt * (omega_b + c * delta_omega_b))` and `c = (m + 1) / (2 * m)` for
`m` simulation substeps. FK and tracking rewards still use world-frame velocities,
with `omega_w = R @ omega_b` at the FK boundary. Motion-file fields ending in `_w`
and MaxCoord retain their world-frame convention.

Network inputs, prediction semantics, parameter shapes, and current-architecture
checkpoints are unchanged. Newly collected replay states use body-frame angular
velocity. Any externally saved GenCoord full states from the previous convention
must first convert their angular block using `omega_b = R.T @ omega_w`; the
identical tensor shape cannot identify the old convention automatically.

### Verification

```bash
uv sync --locked
nvidia-smi
uv run python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
uv run python -m pytest -q
uv run src/supermimic/scripts/check_env.py --device cuda:0
```

`check_env.py` generates a short motion file, runs both tasks with two simulated
robots, checks finite observations/rewards, and backpropagates through the world
model and kinematics. It does not access W&B. Use `--device cpu` for a CPU smoke
check. To inspect the training options:

```bash
uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 --help
```

Training needs either access to the default W&B motion artifact or a local
motion file supplied with `--motion-file /path/to/motion.npz`. For multiple local
clips, pass a quoted glob or comma-separated paths. For local experiments without
W&B uploads, set `WANDB_MODE=disabled` and supply local motion files.

Downloaded datasets, robot assets and converted motion clips belong in the
ignored `data/` directory. They are local inputs, separate from the source
commits; `check_env.py` and the test suite generate their own motion fixtures.
The migrated local example can be used with
`--motion-file data/motions/run1_subject5.npz`.

### Diagnosing startup stalls

The first world-model and policy updates invoke `torch.compile`; initial
compilation can take several minutes before the next learning log appears.
The runner prints the start and return of each first update, and the learner
prints when CUDA graph capture starts and finishes after warmup.

To isolate compilation/capture from simulation and learning, add `--eager True`
to the training command. This disables PyTorch compilation and learner CUDA
graphs, but MuJoCo/Warp still compiles its simulator kernels. For a short check,
also use `--num-envs 128 --runner.wm.batch-size 64
--runner.policy.batch-size 64 --runner.max-iterations 30
--runner.log-interval 1`. Remove `--eager True` to restore accelerated training.

If MuJoCo Warp reports solver or linesearch iteration limits, try
`--env.sim.mujoco.iterations 100 --env.sim.mujoco.ls-iterations 50`.
These increase the solver budget (and can increase simulation cost); they do
not address PyTorch compilation stalls or guarantee convergence for every state.

Checkpoint files are saved locally in all W&B modes. `WANDB_MODE=disabled`
skips artifact uploads and remote pruning entirely. `WANDB_MODE=offline`
queues artifacts for later sync without querying the online registry.

