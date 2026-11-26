"""Synchronized recorded reference/truth/C-posterior overlays using existing ghosts."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import viser
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from mjlab.viewer.viser import MjlabViserScene

from supermimic.validation.config import LOCAL_VARIANTS, VARIANTS


class ReplayRenderer:
  def __init__(self, server, cfg):
    self.cfg = cfg
    self.models = {}
    original = mujoco.MjModel.from_xml_path(str(G1_XML))
    self.joints = np.flatnonzero(original.jnt_type == int(mujoco.mjtJoint.mjJNT_HINGE))
    self.joint_qpos = original.jnt_qposadr[self.joints]
    if len(self.joints) != 29 or original.nq != 36:
      raise ValueError("Replay requires the G1 free-root 29-joint model")
    for layer in ("reference", "truth", "posterior"):
      model = copy.deepcopy(original)
      model.geom_rgba[:] = getattr(cfg, layer + "_rgba")
      # Reuse the existing reference viewer convention: render visual geometry.
      collision = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
      model.geom_rgba[collision, 3] = 0
      self.models[layer] = model
    original.geom_rgba[:, 3] = 0  # Base scene only owns camera/debug plumbing.
    self.scene = MjlabViserScene(server, original, 1)
    self.scene.debug_visualization_enabled = True
    self.data = mujoco.MjData(original)
    mujoco.mj_forward(original, self.data)
    server.scene.add_grid("/floor", width=100, height=100)
    self.variants = (
      LOCAL_VARIANTS if any(v in LOCAL_VARIANTS for v in cfg.methods) else VARIANTS
    )
    for i, variant in enumerate(self.variants):
      server.scene.add_label(
        f"/labels/{variant}", variant, position=(0, i * cfg.panel_spacing, 2)
      )

  def pose_qpos(self, pose, panel):
    qpos = self.data.qpos.copy()
    qpos[:7] = pose[:7]
    qpos[1] += panel * self.cfg.panel_spacing
    qpos[self.joint_qpos] = pose[7:36]
    return qpos

  def draw(self, records, frame, visible):
    self.scene.clear()
    for panel, variant in enumerate(self.variants):
      if variant not in records:
        continue
      for layer, model in self.models.items():
        pose = records[variant][layer][frame]
        if visible[layer] and np.isfinite(pose).all():
          self.scene.add_ghost_mesh(
            self.pose_qpos(pose, panel),
            model,
            alpha=getattr(self.cfg, layer + "_rgba")[3],
            label=f"{variant}/{layer}",
          )
    self.scene.update_from_mjdata(self.data)


def replay_index(output, manifest):
  from supermimic.validation.study import tasks

  index = {}
  for key, motion, variant, _ in tasks(manifest):
    path = output / "tasks" / f"{key}.npz"
    meta = path.with_suffix(".json")
    if not path.exists() or not meta.exists():
      continue
    metadata = json.loads(meta.read_text())
    if metadata["identity"] != manifest["identity"]:
      continue
    with np.load(path, allow_pickle=False) as arrays:
      for row, context in enumerate(arrays["replay_context_ids"]):
        index.setdefault((motion["name"], int(context)), {})[variant] = (key, row)
  return index


def read_replay(output, manifest, entries):
  from supermimic.validation.study import load_task

  result, dt, frames = {}, None, None
  for variant, (key, row) in entries.items():
    task = load_task(output, key, manifest)
    if task is None:
      raise ValueError("Replay checksum mismatch")
    meta, arrays = task
    current = {
      layer: arrays["replay_" + layer][:, row]
      for layer in ("truth", "posterior", "reference")
    }
    if dt is not None and (meta["control_dt"] != dt or len(current["truth"]) != frames):
      raise ValueError("Replay time axes differ between variants")
    dt, frames = meta["control_dt"], len(current["truth"])
    result[variant] = current
  return result, dt, frames


def view(cfg):
  output = Path(cfg.output)
  manifest = json.loads((output / "manifest.json").read_text())
  if manifest["kind"] not in ("joint", "tracking"):
    raise ValueError("Replay requires tracking results")
  index = replay_index(output, manifest)
  if not index:
    raise ValueError("No recorded scenarios available")
  server = viser.ViserServer(port=cfg.viewer_port)
  renderer = ReplayRenderer(server, cfg)
  options = {
    f"{clip} / context {context}": (clip, context) for clip, context in sorted(index)
  }
  selection = server.gui.add_dropdown(
    "Clip / context", options=list(options), initial_value=next(iter(options))
  )
  playing = server.gui.add_checkbox("Play", initial_value=False)
  speed = server.gui.add_number(
    "Speed", initial_value=cfg.playback_speed, min=0.01, max=100, step=0.1
  )
  visibility = {
    layer: server.gui.add_checkbox(layer, initial_value=getattr(cfg, "show_" + layer))
    for layer in ("reference", "truth", "posterior")
  }
  records, dt, frames = read_replay(output, manifest, index[options[selection.value]])
  slider = server.gui.add_slider(
    "Frame", min=0, max=max(1, frames - 1), step=1, initial_value=0
  )
  step = server.gui.add_button("Step")
  info = server.gui.add_markdown(
    "Reference: translucent green; truth: opaque; C posterior: translucent blue-purple. Missing posterior frames are hidden."
  )

  @step.on_click
  def _step(_):
    playing.value = False
    slider.value = min(slider.value + 1, frames - 1)

  previous = selection.value
  deadline = time.monotonic()
  try:
    while True:
      if selection.value != previous:
        playing.value = False
        records, dt, frames = read_replay(
          output, manifest, index[options[selection.value]]
        )
        slider.value = 0
        slider.max = max(1, frames - 1)
        previous = selection.value
      now = time.monotonic()
      if playing.value and now >= deadline:
        slider.value = min(slider.value + 1, frames - 1)
        if slider.value == frames - 1:
          playing.value = False
        deadline = now + dt / speed.value
      frame = min(int(slider.value), frames - 1)
      renderer.draw(records, frame, {k: v.value for k, v in visibility.items()})
      info.content = f"**{previous}** — frame {frame}/{frames - 1}, t={frame * dt:.2f}s. Green: reference; opaque: truth; blue-purple: C posterior. Non-finite posterior frames are hidden."
      time.sleep(min(dt, 0.02))
  except KeyboardInterrupt:
    pass
  finally:
    server.stop()
