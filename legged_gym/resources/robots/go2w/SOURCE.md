# Go2W assets

- Source: [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros/tree/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/go2w_description), commit `5994d4faef0a9cadd3287f8de0199a67eeb2a259`.
- Local changes: relative mesh paths; four wheel collision meshes replaced by cylinders (radius 0.086 m, length 0.045 m, xyz=0, rpy=[π/2,0,0]). Wheel dimensions/orientation follow [TrackinBIT](https://github.com/TrackinBIT/HIMLoco-for-Go2W/tree/011693738c61603c3f22f2bce755098dd36fa7eb). Other explicit dynamics and visual assets remain from the official URDF.
- Active URDF: `urdf/go2w.urdf`, SHA256 `5a6f23475445353158db4396160659f96c5236937c512df91db3332101a0eb91`.
- Active MJCF: `xml/go2w.xml`, generated with `python -m sim2sim.build_model`. It uses explicit inertias and primitive collisions, with scene friction 0.8, self-collisions disabled and timestep 0.002 s.
- `meshes/` contains only the eight files referenced by the active URDF.

History, upstream mass/inertia differences, wheel-axis/importer issues and cleanup decisions: [docs/sim2sim_known_issues.md](../../../../docs/sim2sim_known_issues.md). Historical source copies and removed diagnostics are recoverable from `.cache/sim2sim/cleanup_20261015/before.zip`; current upstream revisions are linked above.

License notices are retained in the existing project license directory: [Unitree MuJoCo](../../../licenses/assets/go2w_unitree_mujoco.txt) and [HIMLoco-for-Go2W](../../../licenses/assets/go2w_HIMLoco_for_Go2W.txt). These notices do not establish a new blanket license for independently supplied URDF/DAE assets.
