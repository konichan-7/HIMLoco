#!/usr/bin/env bash
set -euo pipefail
HIM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HIM_ROOT"
SDK_COMMIT=814556d15970dd2ecf1c9984e845ca02ab07e206
DDS_COMMIT=2cdd114cbd18340c606573b4cc8dc20cc161ec5a
mkdir -p .deps
if [ ! -d .deps/unitree_sdk2_python ]; then
  git clone https://github.com/unitreerobotics/unitree_sdk2_python.git .deps/unitree_sdk2_python
  git -C .deps/unitree_sdk2_python checkout "$SDK_COMMIT"
fi
if [ "$(git -C .deps/unitree_sdk2_python rev-parse HEAD)" != "$SDK_COMMIT" ]; then
  echo 'SDK checkout differs from the pinned version; inspect it before installing.' >&2
  exit 1
fi
if [ ! -d .deps/cyclonedds ]; then
  git clone --depth 1 --branch 0.10.5 https://github.com/eclipse-cyclonedds/cyclonedds.git .deps/cyclonedds
fi
if [ "$(git -C .deps/cyclonedds rev-parse HEAD)" != "$DDS_COMMIT" ]; then
  echo 'Cyclone DDS checkout differs from the pinned version.' >&2
  exit 1
fi
cmake -S .deps/cyclonedds -B .deps/cyclonedds-build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HIM_ROOT/.deps/cyclonedds-install" \
  -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DENABLE_SECURITY=OFF -DENABLE_SSL=OFF
cmake --build .deps/cyclonedds-build --parallel 6
cmake --install .deps/cyclonedds-build
if [ ! -x .venv-sim2sim/bin/python ]; then
  uv venv --python 3.11.16 .venv-sim2sim
fi
export CYCLONEDDS_HOME="$HIM_ROOT/.deps/cyclonedds-install"
uv pip install --python .venv-sim2sim/bin/python -r sim2sim/requirements.lock.txt
.venv-sim2sim/bin/python -m sim2sim.build_model
