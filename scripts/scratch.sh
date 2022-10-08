#!/bin/bash -e
# Continously create images from single prompt
# exploring the options of stable diffusion
#
# Args: Prompt, aspect ratio, scale
# Example:
# ./scripts/explore.sh "Nice view" portrait 9 
BASE_DIR="$(dirname "$(dirname "$(realpath $0)")")"
TEXT_PROMPT="$1"
ASPECT_RATIO="$2"
if [[ -z "$ASPECT_RATIO" ]];
then
  ASPECT_RATIO="portrait"
fi
SCALE="$3"
if [[ -z "$SCALE" ]];
then
  SCALE="7"
fi
echo "Exploring $ASPECT_RATIO images for prompt: '$TEXT_PROMPT' using scale $SCALE"
if [[ "$ASPECT_RATIO" == "portrait" ]];
then
    python "$BASE_DIR/txt2img.py" \
        --prompt "$TEXT_PROMPT" \
        --n_samples 3 \
        --W 448 \
        --H 768 \
        --scales "$SCALE" \
        --outdir "$BASE_DIR/explore/scratch"
fi
if [[ "$ASPECT_RATIO" == "landscape" ]];
then
    python "$BASE_DIR/txt2img.py" \
        --prompt "$TEXT_PROMPT" \
        --n_samples 3 \
        --W 768 \
        --H 448 \
        --scales "$SCALE" \
        --outdir "$BASE_DIR/explore/scratch"
fi
if [[ "$ASPECT_RATIO" == "square" ]];
then
    python "$BASE_DIR/txt2img.py" \
        --prompt "$TEXT_PROMPT" \
        --n_samples 3 \
        --W 512 \
        --H 512 \
        --scales "$SCALE" \
        --outdir "$BASE_DIR/explore/scratch"
fi