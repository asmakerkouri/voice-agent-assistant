#!/usr/bin/env bash
# Downloads the open models (≈ 200 MB) from the sherpa-onnx GitHub releases into ./models
set -euo pipefail
BASE=https://github.com/k2-fsa/sherpa-onnx/releases/download
mkdir -p models && cd models
get() { [ -d "$1" ] || { curl -sSL "$BASE/$2/$1.tar.bz2" | tar xj; echo "  $1"; }; }
get sherpa-onnx-whisper-tiny.en asr-models          # speech-to-text (Whisper tiny.en, int8)
get vits-piper-en_US-amy-low tts-models             # voice of the assistant
for v in en_US-ryan-low en_GB-alan-low en_US-lessac-low; do  # test voices for the benchmark
  get vits-piper-$v tts-models
done
[ -f silero_vad.onnx ] || curl -sSLO "$BASE/asr-models/silero_vad.onnx"  # voice activity detection
echo "models ready in $(pwd)"
