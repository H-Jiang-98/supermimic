"""Map downloaded W&B artifact paths to local files without uploading anything."""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import shlex
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal.datastore import DataStore


def read_records(path: Path) -> Iterator[pb.Record]:
  """Use the installed SDK's framing and checksum validation."""
  store = DataStore()
  try:
    store.open_for_scan(str(path))
    while True:
      offset = store.get_offset()
      data = store.scan_data()
      if data is None:
        # EOF may consume up to six padding bytes, but never a partial record.
        if store.get_offset() - offset > 6:
          raise ValueError(f"Incomplete W&B record: {path}")
        break
      record = pb.Record()
      record.ParseFromString(data)
      yield record
  finally:
    store.close()


def relocate(
  path: Path,
  remote_artifacts: PurePosixPath,
  local_artifacts: Path,
  *,
  dry_run: bool = False,
) -> tuple[int, int]:
  """Validate every referenced file, then atomically replace the local record.

  The first original is retained as ``*.wandb.before-path-map``. Only artifact
  manifest local_path fields change; file digests and all run data are preserved.
  """
  path = path.resolve(strict=True)
  local_artifacts = local_artifacts.resolve(strict=True)
  initial_stat = path.stat()
  replacements: dict[str, str] = {}
  checked: dict[Path, tuple[int, str]] = {}
  finished = False
  for record in read_records(path):
    kind = record.WhichOneof("record_type")
    finished |= kind == "exit"
    if kind != "artifact":
      continue
    for entry in record.artifact.manifest.contents:
      if not entry.local_path:
        continue
      source = PurePosixPath(entry.local_path)
      if source.is_relative_to(remote_artifacts):
        relative = source.relative_to(remote_artifacts)
      elif source.is_relative_to(local_artifacts):
        relative = source.relative_to(local_artifacts)
      else:
        raise ValueError(f"Artifact path is outside configured roots: {source}")
      target = (local_artifacts / relative).resolve()
      if not target.is_relative_to(local_artifacts):
        raise ValueError(f"Artifact path escapes the local directory: {source}")
      if target not in checked:
        if not target.is_file():
          raise FileNotFoundError(f"Missing downloaded artifact: {target}")
        size = target.stat().st_size
        if size != entry.size:
          raise ValueError(f"Artifact size mismatch: {target}")
        with target.open("rb") as stream:
          digest = base64.b64encode(hashlib.file_digest(stream, "md5").digest())
        checked[target] = size, digest.decode("ascii")
      if checked[target] != (entry.size, entry.digest):
        raise ValueError(f"Artifact size/digest mismatch: {target}")
      if str(target) != entry.local_path:
        replacements[entry.local_path] = str(target)
  if not finished:
    raise ValueError(f"Run has no exit record; wait until training finishes: {path}")

  def check_unchanged() -> None:
    current = path.stat()
    if (current.st_ino, current.st_size, current.st_mtime_ns) != (
      initial_stat.st_ino,
      initial_stat.st_size,
      initial_stat.st_mtime_ns,
    ):
      raise RuntimeError(f"W&B file changed during validation: {path}")

  check_unchanged()
  if dry_run or not replacements:
    return len(checked), len(replacements)

  # Keep temporary files on the same filesystem for atomic replacement.
  with tempfile.TemporaryDirectory(prefix=".wandb-path-map-", dir=path.parent) as tmp:
    rewritten = Path(tmp) / "mapped.wandb"
    writer = DataStore()
    try:
      writer.open_for_write(str(rewritten))
      for record in read_records(path):
        if record.WhichOneof("record_type") == "artifact":
          for entry in record.artifact.manifest.contents:
            if entry.local_path in replacements:
              entry.local_path = replacements[entry.local_path]
        writer.write(record)
    finally:
      writer.close()
    check_unchanged()
    backup = path.with_name(path.name + ".before-path-map")
    if not backup.exists():
      snapshot = Path(tmp) / "original.wandb"
      shutil.copy2(path, snapshot)
      check_unchanged()
      # Publish a complete backup, without overwriting an existing one.
      os.link(snapshot, backup)
    shutil.copymode(path, rewritten)
    check_unchanged()
    os.replace(rewritten, path)
  return len(checked), len(replacements)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-dir", type=Path, required=True)
  parser.add_argument("--remote-artifacts-dir", type=PurePosixPath, required=True)
  parser.add_argument("--local-artifacts-dir", type=Path, required=True)
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()
  if not args.remote_artifacts_dir.is_absolute():
    parser.error("--remote-artifacts-dir must be absolute")
  records = sorted(args.run_dir.glob("wandb/offline-run-*/*.wandb"))
  if not records:
    parser.error(f"No wandb/offline-run-*/*.wandb files in {args.run_dir}")
  for path in records:
    print(f"校验离线记录及模型文件：{path}", flush=True)
    count, changed = relocate(
      path, args.remote_artifacts_dir, args.local_artifacts_dir, dry_run=args.dry_run
    )
    action = "待映射" if args.dry_run else "已映射"
    print(f"校验通过：{count} 个模型文件，{action} {changed} 个暂存路径。")
    if not args.dry_run:
      command = (
        f"WANDB_MODE=online uv run wandb sync {shlex.quote(str(path.parent.resolve()))}"
      )
      if Path(str(path) + ".synced").exists():
        command += " --include-synced"
      print(f"本机同步命令（需先登录 W&B）：\n{command}")


if __name__ == "__main__":
  main()
