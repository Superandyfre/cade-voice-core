#!/usr/bin/env bash
# Download ASR / TTS / VAD models for CADE voice pipeline.
#
# Usage: bash scripts/download_voice_models.sh
#
# All models go under /home/pinggu/audio/models/

set -euo pipefail

MODELS_ROOT="/home/pinggu/audio/models"
ASR_DIR="${MODELS_ROOT}/asr"
TTS_DIR="${MODELS_ROOT}/tts"

mkdir -p "${ASR_DIR}" "${TTS_DIR}"

# ------------------------------------------------------------------
# 1. Silero VAD
# ------------------------------------------------------------------
VAD_URL="https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
VAD_PATH="${ASR_DIR}/silero_vad.onnx"

if [ -f "${VAD_PATH}" ]; then
    echo "[skip] silero_vad.onnx already exists"
else
    echo "[download] silero_vad.onnx"
    curl -L -o "${VAD_PATH}" "${VAD_URL}"
fi

# ------------------------------------------------------------------
# 2. Streaming Zipformer English ASR (20M, mobile)
# ------------------------------------------------------------------
ASR_MODEL_NAME="sherpa-onnx-streaming-zipformer-en-20M-2023-02-17-mobile"
ASR_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${ASR_MODEL_NAME}.tar.bz2"
ASR_DEST="${ASR_DIR}/${ASR_MODEL_NAME}"

if [ -d "${ASR_DEST}" ]; then
    echo "[skip] ${ASR_MODEL_NAME} already exists"
else
    echo "[download] ${ASR_MODEL_NAME}"
    curl -L -o "${ASR_DIR}/${ASR_MODEL_NAME}.tar.bz2" "${ASR_URL}"
    tar xjf "${ASR_DIR}/${ASR_MODEL_NAME}.tar.bz2" -C "${ASR_DIR}"
    rm -f "${ASR_DIR}/${ASR_MODEL_NAME}.tar.bz2"
fi

# ------------------------------------------------------------------
# 3. Nemotron 0.6B Streaming ASR (INT8, 560ms chunk)
# ------------------------------------------------------------------
NEMOTRON_MODEL_NAME="sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25"
NEMOTRON_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${NEMOTRON_MODEL_NAME}.tar.bz2"
NEMOTRON_DEST="${ASR_DIR}/${NEMOTRON_MODEL_NAME}"

if [ -d "${NEMOTRON_DEST}" ]; then
    echo "[skip] ${NEMOTRON_MODEL_NAME} already exists"
else
    echo "[download] ${NEMOTRON_MODEL_NAME}"
    curl -L -o "${ASR_DIR}/${NEMOTRON_MODEL_NAME}.tar.bz2" "${NEMOTRON_URL}"
    tar xjf "${ASR_DIR}/${NEMOTRON_MODEL_NAME}.tar.bz2" -C "${ASR_DIR}"
    rm -f "${ASR_DIR}/${NEMOTRON_MODEL_NAME}.tar.bz2"
fi

# ------------------------------------------------------------------
# 4. VITS Piper English TTS (int8 quantised fallback)
# ------------------------------------------------------------------
TTS_MODEL_NAME="vits-piper-en_US-libritts_r-medium-int8"
TTS_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/${TTS_MODEL_NAME}.tar.bz2"
TTS_DEST="${TTS_DIR}/${TTS_MODEL_NAME}"

if [ -d "${TTS_DEST}" ]; then
    echo "[skip] ${TTS_MODEL_NAME} already exists"
else
    echo "[download] ${TTS_MODEL_NAME}"
    curl -L -o "${TTS_DIR}/${TTS_MODEL_NAME}.tar.bz2" "${TTS_URL}"
    tar xjf "${TTS_DIR}/${TTS_MODEL_NAME}.tar.bz2" -C "${TTS_DIR}"
    rm -f "${TTS_DIR}/${TTS_MODEL_NAME}.tar.bz2"
fi

# ------------------------------------------------------------------
# 5. Optional Kokoro / Piper fast TTS models
# ------------------------------------------------------------------
download_optional_tts_model() {
    local name="$1"
    local url="$2"
    local dest="${TTS_DIR}/${name}"
    local archive="${TTS_DIR}/${name}.tar.bz2"

    if [ -d "${dest}" ]; then
        echo "[skip] ${name} already exists"
        return
    fi

    echo "[download optional] ${name}"
    if curl -fL -o "${archive}" "${url}"; then
        tar xjf "${archive}" -C "${TTS_DIR}"
        rm -f "${archive}"
    else
        echo "[warn] optional TTS model unavailable: ${url}"
        rm -f "${archive}"
    fi
}

KOKORO_MODEL_NAME="${KOKORO_MODEL_NAME:-kokoro-en-v0_19}"
KOKORO_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/${KOKORO_MODEL_NAME}.tar.bz2"
download_optional_tts_model "${KOKORO_MODEL_NAME}" "${KOKORO_URL}"

PIPER_FAST_MODEL_NAME="${PIPER_FAST_MODEL_NAME:-vits-piper-en_US-lessac-medium}"
PIPER_FAST_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/${PIPER_FAST_MODEL_NAME}.tar.bz2"
download_optional_tts_model "${PIPER_FAST_MODEL_NAME}" "${PIPER_FAST_URL}"

# ------------------------------------------------------------------
# 6. Verify critical files
# ------------------------------------------------------------------
echo ""
echo "=== Verification ==="

ERRORS=0

check_file() {
    if [ -f "$1" ]; then
        echo "  OK  $1"
    else
        echo "  MISSING  $1"
        ERRORS=$((ERRORS + 1))
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo "  OK  $1/"
    else
        echo "  MISSING  $1/"
        ERRORS=$((ERRORS + 1))
    fi
}

# VAD
check_file "${VAD_PATH}"

# ASR — look for encoder, decoder, joiner, tokens
check_dir "${ASR_DEST}"
for f in encoder decoder joiner tokens; do
    MATCH=$(find "${ASR_DEST}" -name "*${f}*" -type f 2>/dev/null | head -1)
    if [ -n "${MATCH}" ]; then
        echo "  OK  ${MATCH}"
    else
        echo "  MISSING  ${ASR_DEST}/*${f}*"
        ERRORS=$((ERRORS + 1))
    fi
done

# Nemotron — look for encoder, decoder, joiner, tokens
check_dir "${NEMOTRON_DEST}"
for f in encoder decoder joiner tokens; do
    MATCH=$(find "${NEMOTRON_DEST}" -name "*${f}*" -type f 2>/dev/null | head -1)
    if [ -n "${MATCH}" ]; then
        echo "  OK  ${MATCH}"
    else
        echo "  MISSING  ${NEMOTRON_DEST}/*${f}*"
        ERRORS=$((ERRORS + 1))
    fi
done

# TTS — look for .onnx and tokens.txt
check_dir "${TTS_DEST}"
TTS_ONNX=$(find "${TTS_DEST}" -name "*.onnx" -type f 2>/dev/null | head -1)
if [ -n "${TTS_ONNX}" ]; then
    echo "  OK  ${TTS_ONNX}"
else
    echo "  MISSING  ${TTS_DEST}/*.onnx"
    ERRORS=$((ERRORS + 1))
fi
if [ -f "${TTS_DEST}/tokens.txt" ]; then
    echo "  OK  ${TTS_DEST}/tokens.txt"
else
    echo "  MISSING  ${TTS_DEST}/tokens.txt"
    ERRORS=$((ERRORS + 1))
fi

# Optional Kokoro — verify only when downloaded
KOKORO_DEST="${TTS_DIR}/${KOKORO_MODEL_NAME}"
if [ -d "${KOKORO_DEST}" ]; then
    check_dir "${KOKORO_DEST}"
    for f in model.onnx voices.bin tokens.txt; do
        check_file "${KOKORO_DEST}/${f}"
    done
fi

# Optional Piper fast — verify only when downloaded
PIPER_FAST_DEST="${TTS_DIR}/${PIPER_FAST_MODEL_NAME}"
if [ -d "${PIPER_FAST_DEST}" ]; then
    check_dir "${PIPER_FAST_DEST}"
    PIPER_FAST_ONNX=$(find "${PIPER_FAST_DEST}" -name "*.onnx" -type f 2>/dev/null | head -1)
    if [ -n "${PIPER_FAST_ONNX}" ]; then
        echo "  OK  ${PIPER_FAST_ONNX}"
    else
        echo "  MISSING  ${PIPER_FAST_DEST}/*.onnx"
        ERRORS=$((ERRORS + 1))
    fi
    check_file "${PIPER_FAST_DEST}/tokens.txt"
fi

echo ""
if [ "${ERRORS}" -gt 0 ]; then
    echo "FAILED: ${ERRORS} file(s) missing"
    exit 1
else
    echo "All model files verified."
fi
