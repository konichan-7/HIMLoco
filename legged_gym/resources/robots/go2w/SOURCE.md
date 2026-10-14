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

## Migration checks

The normalized URDF matches the complete original element/attribute structure after substituting the mesh URLs; all 8 distinct referenced meshes resolve. All user-provided and official meshes retain their bytes. The original and relocated official MJCF compile to identical mass, inertia, transforms, joint axes/ranges, damping, armature, contact parameters, actuator limits/gears and mesh geometry. The runtime scene rebuilt successfully and all 5 existing sim2sim contract tests passed. The structural/asset check report is saved locally at `.cache/sim2sim/resource_migration/validation.json`.
