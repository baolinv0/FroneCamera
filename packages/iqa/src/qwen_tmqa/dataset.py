from __future__ import annotations

import re
from pathlib import Path

from .config import DatasetConfig
from .domain import AlphaImage, SceneSpec

_LEVEL_PATTERN = re.compile(r"^a_(?P<sign>[mp])(?P<magnitude>\d{3})$")
_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_alpha_level(level: str) -> float:
    if level == "a_000":
        return 0.0
    match = _LEVEL_PATTERN.match(level)
    if not match:
        raise ValueError(f"unsupported alpha level: {level}")
    value = int(match.group("magnitude")) / 100.0
    return -value if match.group("sign") == "m" else value


def _image_relatives(level_dir: Path, recursive: bool) -> set[Path]:
    candidates = level_dir.rglob("*") if recursive else level_dir.glob("*")
    return {
        path.relative_to(level_dir)
        for path in candidates
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    }


def discover_scenes(root: Path, config: DatasetConfig) -> list[SceneSpec]:
    if not root.exists():
        raise FileNotFoundError(root)

    level_dirs: dict[str, Path] = {}
    illegal_alpha_dirs: list[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not child.name.startswith("a_"):
            continue
        try:
            parse_alpha_level(child.name)
        except ValueError:
            illegal_alpha_dirs.append(child.name)
            continue
        level_dirs[child.name] = child

    expected = list(config.expected_levels)
    expected_set = set(expected)
    discovered_set = set(level_dirs)
    if config.strict_complete:
        if illegal_alpha_dirs:
            raise ValueError(f"illegal alpha level directories: {sorted(illegal_alpha_dirs)}")
        missing_dirs = sorted(expected_set - discovered_set)
        if missing_dirs:
            raise ValueError(f"missing expected level directories: {missing_dirs}")
        unknown_dirs = sorted(discovered_set - expected_set)
        if unknown_dirs:
            raise ValueError(f"unknown alpha level directories: {unknown_dirs}")

    if config.baseline_level not in level_dirs:
        raise ValueError(f"baseline level {config.baseline_level} not found under {root}")

    active_levels = expected if config.strict_complete else sorted(
        level_dirs,
        key=parse_alpha_level,
    )
    relative_by_level = {
        level: _image_relatives(level_dirs[level], config.recursive)
        for level in active_levels
        if level in level_dirs
    }
    baseline_relatives = relative_by_level[config.baseline_level]
    scene_relatives = (
        set().union(*relative_by_level.values()) if config.strict_complete else baseline_relatives
    )

    scenes: list[SceneSpec] = []
    for relative in sorted(scene_relatives, key=lambda item: item.as_posix()):
        missing = [
            level
            for level in active_levels
            if level not in relative_by_level or relative not in relative_by_level[level]
        ]
        if config.strict_complete and missing:
            raise ValueError(f"scene {relative} missing levels: {sorted(missing)}")
        if relative not in baseline_relatives:
            if config.strict_complete:
                raise ValueError(f"scene {relative} missing baseline level {config.baseline_level}")
            continue

        alpha_images = [
            AlphaImage(
                level=level,
                alpha=parse_alpha_level(level),
                path=level_dirs[level] / relative,
            )
            for level in active_levels
            if level in relative_by_level and relative in relative_by_level[level]
        ]
        scenes.append(
            SceneSpec(
                scene_id=relative.as_posix(),
                baseline_path=level_dirs[config.baseline_level] / relative,
                alpha_images=alpha_images,
                generator=config.generator,
            )
        )
    return scenes
