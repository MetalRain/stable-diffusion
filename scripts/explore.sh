#!/bin/bash -e
# Continously create images from single prompt
# exploring the options of stable diffusion
#
# Args: Prompt, aspect ratio, scale
# Example:
# ./scripts/explore.sh "Nice view" portrait 9 
BASE_DIR="$(dirname "$(dirname "$(realpath $0)")")"
source "$BASE_DIR/scripts/config.sh"

TEXT_PROMPT="$1"
ASPECT_RATIO="$2"
if [[ -z "$ASPECT_RATIO" ]];
then
  ASPECT_RATIO="$DEFAULT_ASPECT_RATIO"
fi
SCALE="$3"
if [[ -z "$SCALE" ]];
then
  SCALE="$DEFAULT_SCALE"
fi

echo "Generating $ASPECT_RATIO for prompt: '$TEXT_PROMPT' using scale $SCALE"
PROMPT_HASH=$(bash $BASE_DIR/scripts/init-explore.sh "$TEXT_PROMPT")
echo "Images will be in $BASE_DIR/explore/$PROMPT_HASH/"
if [[ "$ASPECT_RATIO" == "portrait" ]];
then
    python "$BASE_DIR/main.py" \
        --prompt "$TEXT_PROMPT" \
        --n_samples "$MAX_EXPLORE_ITERATIONS" \
        --W "$MIN_RECT_DIM" \
        --H "$MAX_RECT_DIM" \
        --scales "$SCALE" \
        --task "txt2img" \
        --plms "1" \
        --outdir "$BASE_DIR/explore/$PROMPT_HASH"
fi
if [[ "$ASPECT_RATIO" == "landscape" ]];
then
    python "$BASE_DIR/main.py" \
        --prompt "$TEXT_PROMPT" \
        --n_samples "$MAX_EXPLORE_ITERATIONS" \
        --W "$MAX_RECT_DIM" \
        --H "$MIN_RECT_DIM" \
        --scales "$SCALE" \
        --task "txt2img" \
        --plms "1" \
        --outdir "$BASE_DIR/explore/$PROMPT_HASH"
fi
if [[ "$ASPECT_RATIO" == "square" ]];
then
    python "$BASE_DIR/main.py" \
        --prompt "$TEXT_PROMPT" \
        --n_samples "$MAX_EXPLORE_ITERATIONS" \
        --W "$MAX_SQUARE_DIM" \
        --H "$MAX_SQUARE_DIM" \
        --scales "$SCALE" \
        --task "txt2img" \
        --plms "1" \
        --outdir "$BASE_DIR/explore/$PROMPT_HASH"
fi