"""Isaac Gym training height samples, adapted to MuJoCo terrain colliders.

Height generators adapted from NVIDIA terrain_utils (BSD-3-Clause).
Source and license: legged_gym/resources/licenses/isaacgym_terrain.txt.
"""

import ast
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np

from .config import ROOT

CONFIG = ROOT / "legged_gym/legged_gym/envs/base/legged_robot_config.py"
GENERATOR = ROOT / "legged_gym/legged_gym/utils/terrain.py"
TRAINING_KINDS = ("slope", "rough_slope", "stairs_down", "stairs_up", "obstacles")
KINDS = ("flat", "mixed", "waves", *TRAINING_KINDS)


def settings():
    tree = ast.parse(CONFIG.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LeggedRobotCfg"
    )
    terrain = next(
        n for n in cls.body if isinstance(n, ast.ClassDef) and n.name == "terrain"
    )
    return {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in terrain.body
        if isinstance(n, ast.Assign)
    }


def height_samples(kind, difficulty, cfg, rng):
    """Use the training generator's discretization, including int truncation."""
    hs, vs = cfg["horizontal_scale"], cfg["vertical_scale"]
    nx, ny = int(cfg["terrain_length"] / hs), int(cfg["terrain_width"] / hs)
    h = np.zeros((nx, ny), dtype=np.int16)
    if kind == "waves":
        # Local sim2sim addition: smooth 2-D rolling terrain, not a training claim.
        x = np.arange(nx)[:, None] * hs
        y = np.arange(ny)[None, :] * hs
        envelope = (
            np.sin(np.pi * x / cfg["terrain_length"]) ** 2
            * np.sin(np.pi * y / cfg["terrain_width"]) ** 2
        )
        amplitude = 0.08 + 0.08 * difficulty
        surface = (
            amplitude
            * envelope
            * (np.sin(2 * np.pi * x / 2) + 0.35 * np.sin(2 * np.pi * y / 3))
            / 1.35
        )
        h = np.rint(surface / vs).astype(np.int16)
    elif kind in ("slope", "rough_slope"):
        x = (nx // 2 - np.abs(nx // 2 - np.arange(nx))) / (nx // 2)
        y = (ny // 2 - np.abs(ny // 2 - np.arange(ny))) / (ny // 2)
        max_height = int(difficulty * 0.4 * hs / vs * (nx / 2))
        h += (max_height * x[:, None] * y[None, :]).astype(np.int16)
        edge = h[nx // 2 - int(3 / hs / 2), ny // 2 - int(3 / hs / 2)]
        h = np.clip(h, min(edge, 0), max(edge, 0))
        if kind == "rough_slope":
            amplitude = 0.01 + 0.07 * difficulty
            low, high = int(-amplitude / vs), int(amplitude / vs)
            coarse = rng.choice(
                np.arange(low, high + 1), (int(nx * hs / 0.2), int(ny * hs / 0.2))
            )
            # Preview 4 interp2d(kind='linear'), expressed without a SciPy dependency.
            x = np.linspace(0, nx * hs, coarse.shape[0])
            y = np.linspace(0, ny * hs, coarse.shape[1])
            fine_x, fine_y = np.linspace(0, nx * hs, nx), np.linspace(0, ny * hs, ny)
            along_y = np.array([np.interp(fine_y, y, row) for row in coarse])
            fine = np.array([np.interp(fine_x, x, col) for col in along_y.T]).T
            h += np.rint(fine).astype(np.int16)
    elif kind.startswith("stairs_"):
        step = int(0.30 / hs)
        height = int((0.05 + 0.18 * difficulty) / vs)
        if kind == "stairs_down":
            height *= -1
        start, stop, z = step, nx - step, height
        while stop - start + 2 * step > int(3 / hs):
            h[start:stop, start:stop] = z
            start, stop, z = start + step, stop - step, z + height
    elif kind == "obstacles":
        maximum = int((0.05 + difficulty * 0.1) / vs)
        for _ in range(20):
            width, length = rng.choice(range(int(1 / hs), int(2 / hs), 4), 2)
            x = rng.choice(range(0, nx - width, 4))
            y = rng.choice(range(0, ny - length, 4))
            h[x : x + width, y : y + length] = rng.choice(
                [-maximum, -maximum // 2, maximum // 2, maximum]
            )
        platform = int(3 / hs)
        h[
            (nx - platform) // 2 : (nx + platform) // 2,
            (ny - platform) // 2 : (ny + platform) // 2,
        ] = 0
    return h


def rectangles(samples):
    """Cover equal-height regions with non-overlapping boxes, including pits."""
    pending = np.ones(samples.shape, dtype=bool)
    for x in range(samples.shape[0]):
        for y in range(samples.shape[1]):
            if not pending[x, y]:
                continue
            z = samples[x, y]
            y2 = y + 1
            while y2 < samples.shape[1] and pending[x, y2] and samples[x, y2] == z:
                y2 += 1
            x2 = x + 1
            while x2 < samples.shape[0] and np.all(
                pending[x2, y:y2] & (samples[x2, y:y2] == z)
            ):
                x2 += 1
            pending[x:x2, y:y2] = False
            yield x, x2, y, y2, z


def build_scene(model_path, output, kind, level=0, seed=0):
    """Save a reproducible scene; leave robot assets and controller untouched."""
    cfg = settings()
    root = ET.parse(model_path).getroot()
    for asset in root.findall(".//asset/*[@file]"):
        asset.set("file", str((model_path.parent / asset.get("file")).resolve()))
    world = root.find("worldbody")
    floor = world.find("geom[@name='floor']")
    world.remove(floor)
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    length, width = cfg["terrain_length"], cfg["terrain_width"]
    hs, vs = cfg["horizontal_scale"], cfg["vertical_scale"]
    rows, cols = (3, 2) if kind == "mixed" else (1, 1)
    difficulty = level / cfg["num_rows"]
    rng = np.random.RandomState(seed)
    tiles = []
    palettes = {
        "flat": "0.25 0.29 0.32 1",
        "waves": "0.42 0.56 0.42 1",
        "slope": "0.42 0.56 0.42 1",
        "rough_slope": "0.48 0.50 0.36 1",
        "stairs_down": "0.40 0.50 0.66 1",
        "stairs_up": "0.65 0.51 0.33 1",
        "obstacles": "0.54 0.43 0.55 1",
    }
    common = {
        "friction": f"{cfg['static_friction']} 0 0",
        "condim": "3",
        "contype": "2",
        "conaffinity": "1",
        "group": "0",
    }
    for col in range(cols):
        for row in range(rows):
            tile_kind = (
                ("flat", "waves", "stairs_up" if col == 0 else "stairs_down")[row]
                if kind == "mixed"
                else kind
            )
            samples = height_samples(tile_kind, difficulty, cfg, rng)
            x0, y0 = row * length, col * width
            name = f"terrain_{row}_{col}"
            color = palettes[tile_kind]
            if tile_kind.startswith("stairs_") or tile_kind in ("obstacles", "flat"):
                bottom = float(samples.min() * vs) - 0.2
                for i, (x1, x2, y1, y2, z) in enumerate(rectangles(samples)):
                    top = float(z * vs)
                    ET.SubElement(
                        world,
                        "geom",
                        name=f"{name}_{i}",
                        type="box",
                        pos=f"{x0 + (x1 + x2) * hs / 2} {y0 + (y1 + y2) * hs / 2} {(top + bottom) / 2}",
                        size=f"{(x2 - x1) * hs / 2} {(y2 - y1) * hs / 2} {(top - bottom) / 2}",
                        rgba=color,
                        **common,
                    )
            else:
                padded = np.pad(samples, ((0, 1), (0, 1)), mode="edge") * vs
                low, high = float(padded.min()), float(padded.max())
                span = max(high - low, vs)
                ET.SubElement(
                    asset,
                    "hfield",
                    name=name,
                    nrow=str(padded.shape[1]),
                    ncol=str(padded.shape[0]),
                    size=f"{length / 2} {width / 2} {span} 0.2",
                    elevation=" ".join(
                        f"{v:.7g}" for v in ((padded.T[::-1] - low) / span).flat
                    ),
                )
                ET.SubElement(
                    world,
                    "geom",
                    name=name,
                    type="hfield",
                    hfield=name,
                    pos=f"{x0 + length / 2} {y0 + width / 2} {low}",
                    rgba=color,
                    **common,
                )
            center = samples[
                int((length / 2 - 1) / hs) : int((length / 2 + 1) / hs),
                int((width / 2 - 1) / hs) : int((width / 2 + 1) / hs),
            ]
            origin = [x0 + length / 2, y0 + width / 2, float(center.max() * vs)]
            tiles.append(
                {
                    "row": row,
                    "col": col,
                    "kind": tile_kind,
                    "difficulty": difficulty,
                    "min_height_m": float(samples.min() * vs),
                    "max_height_m": float(samples.max() * vs),
                    "origin": origin,
                    "sample_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
                }
            )
    spawn = tiles[0]["origin"]
    # Training's surrounding border is flat; no infinite plane may fill stair pits.
    border = cfg["border_size"]
    lx, ly = rows * length, cols * width
    for i, (x, y, sx, sy) in enumerate(
        (
            (-border / 2, ly / 2, border / 2, ly / 2 + border),
            (lx + border / 2, ly / 2, border / 2, ly / 2 + border),
            (lx / 2, -border / 2, lx / 2, border / 2),
            (lx / 2, ly + border / 2, lx / 2, border / 2),
        )
    ):
        ET.SubElement(
            world,
            "geom",
            name=f"border_{i}",
            type="box",
            pos=f"{x} {y} -0.1",
            size=f"{sx} {sy} 0.1",
            material="ground",
            **common,
        )
    statistic = root.find("statistic")
    if statistic is None:
        statistic = ET.SubElement(root, "statistic")
    # Keep near clipping at robot scale.
    statistic.set("extent", "2")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    view_map = visual.find("map")
    if view_map is None:
        view_map = ET.SubElement(visual, "map")
    view_map.set("znear", "0.01")
    view_map.set("zfar", "300")
    ET.indent(root)
    output.write_text(ET.tostring(root, encoding="unicode"))
    info = {
        "kind": kind,
        "level": level,
        "seed": seed,
        "spawn": spawn,
        "settings": cfg,
        "tiles": tiles,
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "training_generator_sha256": hashlib.sha256(GENERATOR.read_bytes()).hexdigest(),
        "collision": "boxes for stairs/obstacles; heightfields for slopes",
    }
    output.with_suffix(".terrain.json").write_text(json.dumps(info, indent=2) + "\n")
    return info
