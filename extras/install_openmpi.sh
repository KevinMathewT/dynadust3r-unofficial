#!/usr/bin/env bash
set -euo pipefail

# ───────── CONFIG ─────────
OMPI_VERSION="4.1.5"
PREFIX="$HOME/opt/openmpi"
NUM_THREADS="$(nproc)"
# capture the directory where you launched this script
PROJECT_DIR="$(pwd)"
# ──────────────────────────

echo "📥 Downloading Open MPI v$OMPI_VERSION..."
TMPDIR=$(mktemp -d)
cd "$TMPDIR"
wget -q "https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OMPI_VERSION}.tar.gz"

echo "📦 Extracting..."
tar xf "openmpi-${OMPI_VERSION}.tar.gz"
cd "openmpi-${OMPI_VERSION}"

echo "⚙️  Configuring (prefix=$PREFIX)..."
./configure --prefix="$PREFIX" --disable-silent-rules \
            CFLAGS="-O3" CXXFLAGS="-O3" FCFLAGS="-O3"

echo "🔨 Building with $NUM_THREADS threads..."
make -j "$NUM_THREADS"

echo "📥 Installing to $PREFIX..."
make install

echo "✅ Open MPI $OMPI_VERSION installed."

echo "🚀 Updating PATH and LD_LIBRARY_PATH for this session..."
export PATH="$PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH"

echo " which mpicc: $(which mpicc)"
echo " which mpirun: $(which mpirun)"
echo "→ To persist, add these lines to your ~/.bashrc or ~/.zshrc:"
echo "   export PATH=\"$PREFIX/bin:\$PATH\""
echo "   export LD_LIBRARY_PATH=\"$PREFIX/lib:\$LD_LIBRARY_PATH\""

# ───────── Move back and install mpi4py ─────────
echo "🐍 Installing mpi4py into your project’s Poetry environment…"
cd "$PROJECT_DIR"

# ensure Poetry sees its pyproject.toml
if [ ! -f "pyproject.toml" ]; then
  echo "❌ pyproject.toml not found in $PROJECT_DIR"
  exit 1
fi

HDF5_MPI="ON" MPICC=mpicc poetry run pip install --no-binary=h5py --force-reinstall h5py

echo "✅ mpi4py installed in Poetry."

# ───────── Cleanup ─────────
rm -rf "$TMPDIR"
echo "🎉 Done! You can now run your MPI-enabled script with:  mpirun -n 8 poetry run python -m datasets_preprocess.preprocess_stereo4d_parallel_hdf5"
