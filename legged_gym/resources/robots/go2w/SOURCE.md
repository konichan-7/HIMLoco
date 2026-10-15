# Go2W robot resources

Normalized on 2026-10-13 to match the other standalone robot resources in this repository:

```text
go2w/
  urdf/go2w.urdf
  meshes/
  xml/go2w.xml
  LICENSE.unitree_mujoco
  SOURCE.md
```

## URDF

The user-provided `go2w_description` ROS description was relocated from `urdf/go2w_description.urdf` to `urdf/go2w.urdf`. Meshes moved from `dae/` to `meshes/`; `package://go2w_description/dae/` URLs became `../meshes/` paths. XML whitespace was normalized. Joint/link names, masses, inertias, transforms, collision geometry, joint types, limits and transmission data were preserved.

- Original URDF SHA256: `ff16b794d7987ef16ffb34d4bd56ae5a911ea142155fcb596e31b35a385a4322`
- Normalized URDF SHA256: `a9021c08c7ddaeaac561384914092959e18f2f3d6af77c15dba536ad8af87267`
- Original description mass: approximately 19.523 kg.

The original source repository/commit of this user-provided description has not been established. `LICENSE.unitree_mujoco` describes the separately vendored official MJCF/assets below, not a newly established license for all user-provided URDF/DAE files.

ROS-only `CMakeLists.txt`, `package.xml`, `launch/` and `config/` were removed from the active standalone resource tree. The complete pre-migration Go2W resources and the former sim2sim official-model tree are backed up locally in `.cache/sim2sim/resource_migration/go2w_before.zip` (ignored by Git).

## Official MuJoCo model

Vendored from [unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco/tree/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/unitree_robots/go2w), commit `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d`.

- Upstream file: `unitree_robots/go2w/go2w.xml`
- Upstream MJCF SHA256: `c8feaef4afdf360335727c80a826d1611950c562a3daaa5b5bfcf8b57f6859a6`
- Relocated MJCF SHA256: `397c0d94c15c128ed99b3d84b186803bc43184ab92a81d0fe25b33f43c50703e`
- Sole change to this input MJCF: `meshdir="assets"` → `meshdir="../meshes"`.
- Its 17 referenced mesh files are copied without modification. `calf.stl` and `calf_mirror.stl` were identical to the URDF meshes and are shared; all original URDF meshes are retained.
- Compiled official-model mass: approximately 19.126408 kg, with 23 qpos / 22 qvel / 16 actuators.
- Official MJCF/mesh license: [LICENSE.unitree_mujoco](LICENSE.unitree_mujoco).

`sim2sim.build_model` derives `.cache/sim2sim/go2w/scene.xml` from this canonical MJCF. It adapts joint/actuator names, control limits, the 2 ms timestep and integrator, adds ground/light and renames the accelerometer sensor. It does not modify the canonical resource files; the generated manifest records source, referenced asset and runtime model hashes.

This is an asset-layout migration. The URDF and official MJCF retain their original physical differences; Isaac Gym loading and cross-engine dynamics equivalence still require validation on the training machine.

## 2026-10-14 Isaac Gym calf visual correction

The four calf visual references now use `calf.dae` / `calf_mirror.dae`. Both files were copied byte-for-byte from the user-specified sibling checkout of [TrackinBIT/HIMLoco-for-Go2W](https://github.com/TrackinBIT/HIMLoco-for-Go2W/tree/011693738c61603c3f22f2bce755098dd36fa7eb), commit `011693738c61603c3f22f2bce755098dd36fa7eb`, paths `resources/robots/go2w/dae/calf.dae` and `calf_mirror.dae`. The source repository's license notice is retained as `LICENSE.HIMLoco_for_Go2W`; this does not establish a separate original asset-author license. The original STL files remain available for the official MJCF.

- `calf.dae` SHA256: `29872fa1a435b92bc2b395fdfa7f0aee26101db3c494681896bd4ef24ee7883c`
- `calf_mirror.dae` SHA256: `79c66be7ed06760bd5001c067a1fb0ac517f639d8b1a5f89be471903b3f359e1`
- Updated URDF SHA256: `aa5d9539e6c20e91a74cfc46430b6f4945bf6030da190818c8614f92da38cd2e`

Actual Isaac Gym Preview 4 visual/collision camera renders confirmed that the former STL calf visuals were rotated incorrectly with this mixed DAE/STL asset and `flip_visual_attachments=True`, making the wheels appear disconnected. Using the reference DAE calf visuals restores the visible connections while preserving that importer setting for the existing base/hip/thigh/wheel DAE meshes. Disabling the setting globally rendered those other meshes incorrectly.

Only the four calf `visual` mesh filenames changed. After removing `visual` elements, the before/after URDF trees are byte-identical; camera images rendered with `use_collision_geometry=True` are pixel-identical. Joint transforms/types/limits, masses, inertias, collisions, wheel meshes and training controls are unchanged. Native Gym reports all four existing continuous wheel joints as `hasLimits=False`; replacing them with the reference's large-limit revolute joints is unnecessary on this installation.

Evidence and the sibling comparison are in `outputs/go2w_urdf_review_20261014/`. The prior migration hashes above describe the historical, pre-correction resources. This visual correction does not resolve the pre-existing URDF/MJCF physical differences or constitute closed-loop sim2sim acceptance.

## Migration checks

The normalized URDF matches the complete original element/attribute structure after substituting the mesh URLs; all 8 distinct referenced meshes resolve. All user-provided and official meshes retain their bytes. The original and relocated official MJCF compile to identical mass, inertia, transforms, joint axes/ranges, damping, armature, contact parameters, actuator limits/gears and mesh geometry. The runtime scene rebuilt successfully and all 5 existing sim2sim contract tests passed. The structural/asset check report is saved locally at `.cache/sim2sim/resource_migration/validation.json`.
