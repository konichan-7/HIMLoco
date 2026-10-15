# Go2W assets

- Source: [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros/tree/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/go2w_description), commit `5994d4faef0a9cadd3287f8de0199a67eeb2a259`.
- Local changes: relative mesh paths; four wheel collision meshes replaced by cylinders (radius 0.086 m, length 0.045 m, xyz=0, rpy=[π/2,0,0]). Wheel dimensions/orientation follow [TrackinBIT](https://github.com/TrackinBIT/HIMLoco-for-Go2W/tree/011693738c61603c3f22f2bce755098dd36fa7eb). Other explicit dynamics and visual assets remain from the official URDF.
- Active URDF: `urdf/go2w.urdf`, SHA256 `ec99cb80627ce147be4679ad2c9dd6e81ac387a0e8b8e77cdc20f4dd747767b5`.
- Active MJCF: `xml/go2w.xml`, generated with `python -m sim2sim.build_model`. It uses explicit inertias and primitive collisions, with scene friction 0.8, self-collisions disabled and timestep 0.002 s.
- `meshes/` contains the eight files referenced by the active URDF, plus the two original calf STL files retained as sources for their equivalent DAE representations.

History, upstream mass/inertia differences, wheel-axis/importer issues and cleanup decisions: [docs/sim2sim_known_issues.md](../../../../docs/sim2sim_known_issues.md). Historical source copies and removed diagnostics are recoverable from `.cache/sim2sim/cleanup_20261015/before.zip`; current upstream revisions are linked above.

License notices are retained in the existing project license directory: [Unitree MuJoCo](../../../licenses/assets/go2w_unitree_mujoco.txt) and [HIMLoco-for-Go2W](../../../licenses/assets/go2w_HIMLoco_for_Go2W.txt). These notices do not establish a new blanket license for independently supplied URDF/DAE assets.

Isaac Gym Preview 4 compatibility: four wheel joints use `revolute` with ±999999 rad bounds; wheel control remains velocity-only (`kp=0`). Earlier CPU/GPU torque-response tests required this importer workaround. All upstream cylinder collisions, masses, inertias, transforms, effort/velocity limits and visuals are retained. This updated asset requires fresh regression and learning validation; previous mesh-wheel results do not establish acceptance. Evidence: `outputs/go2w_watchdog_20261015/post_pull_b6e3bb2/`.

## 2026-10-15 Isaac Gym calf visual compatibility

The four calf visual references use `calf.dae` / `calf_mirror.dae`, generated directly from the pinned official `calf.stl` / `calf_mirror.stl` above. They are **not** the historical calf DAE copies from HIMLoco-for-Go2W. Each triangle corner and normal round-trips exactly as float32; the files use metre units, Z_UP and an identity scene node, with no scaling, translation or geometry change. The STL originals remain unchanged.

This representation fixes Isaac Gym Preview4's mixed STL/DAE visual-axis import with `flip_visual_attachments=True`, preserving the shared URDF visual coordinate frame. The global flip option, four visual origins, all collisions, joints and inertial data remain unchanged. No training or simulator implementation changes are needed.

| Representation | Source STL SHA256 | Generated DAE SHA256 |
|---|---|---|
| calf | `2aee434b80da010deebb11c8538d593f7901a169cc34916350902746e591437e` | `3ed2c57cbd610adfd3edf98574a98e28508aebfba67b671626fac318dbf384f4` |
| calf_mirror | `528b471bfc60b6e72dbb162b34eefebced994bea82aaf58eda91f090219cd911` | `1bcd9e334f5a32270e5cfbb525909cd850cb158767e7842fd496bbeea532ecb1` |

Verification: nonvisual URDF elements identical; 500-step CPU PhysX trajectories and same-coordinate collision images identical; explicit `python -m sim2sim.build_model` rebuild byte-identical. Current headless training continues with its originally loaded visual asset; its startup asset hash remains an accurate historical record. New viewers load the repaired visuals. Conversion command/evidence: `outputs/go2w_render_fix_20261015/convert_calf_visuals.py` and `production_verification.json`.
