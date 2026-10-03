#!/usr/bin/env bash
# Builds win_player.exe for Windows x86_64 from WSL or Linux and copies it to bin/sidecar/win_player.exe, the
# prebuilt copy the plugin ships (the sidecar stages it into %LOCALAPPDATA%\live-vibe). Also builds the host binary,
# which the sidecar's --unit checks run with --fake. Commit bin/sidecar/win_player.exe after a rebuild.
#
# Needs: rustup target add x86_64-pc-windows-gnu; sudo apt install gcc-mingw-w64-x86-64 (the linker).
# The exe links only Windows system DLLs, and should run on Windows 11 on Arm through x64 emulation (untested).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"
target=x86_64-pc-windows-gnu
cargo test --quiet
cargo build --release --quiet
# no local paths in the shipped binary (panic locations carry source paths)
RUSTFLAGS="--remap-path-prefix=$here=win_player --remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}=cargo" \
  cargo build --release --quiet --target "$target"
cp "target/$target/release/win_player.exe" ../bin/sidecar/win_player.exe
echo "bin/sidecar/win_player.exe: $(stat -c %s ../bin/sidecar/win_player.exe) bytes," \
  "sha256 $(sha256sum ../bin/sidecar/win_player.exe | cut -c1-12)"
