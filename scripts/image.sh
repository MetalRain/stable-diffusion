#!/bin/bash -e
# Make single image
#
# Args: Prompt, aspect ratio, seed, scale
# Example:
# ./scripts/image.sh "Nice view" portrait 2142523 9
BASE_DIR="$(dirname "$(dirname "$(realpath $0)")")"
source "$BASE_DIR/scripts/config.sh"

TEXT_PROMPT="$1"
ASPECT_RATIO="$2"
if [[ -z "$ASPECT_RATIO" ]];
then
  ASPECT_RATIO="$DEFAULT_ASPECT_RATIO"
fi
SEED="$3"
SCALE="$4"
if [[ -z "$SCALE" ]];
then
  SCALE="$DEFAULT_SCALE"
fi
echo "Generating $ASPECT_RATIO for prompt: '$TEXT_PROMPT' with seed $SEED and scale $SCALE"
PROMPT_HASH=$(bash $BASE_DIR/scripts/init-explore.sh "$TEXT_PROMPT")
echo "Image will be in $BASE_DIR/explore/$PROMPT_HASH/"
if [[ "$ASPECT_RATIO" == "portrait" ]];
then
    python "$BASE_DIR/main.py" \
        --prompt "$TEXT_PROMPT" \
        --seed "$SEED" \
        --n_samples 1 \
        --W "$MIN_RECT_DIM" \
        --H "$MAX_RECT_DIM" \
        --scales "$SCALE" \
        --outdir "$BASE_DIR/explore/$PROMPT_HASH"
fi
if [[ "$ASPECT_RATIO" == "landscape" ]];
then
    python "$BASE_DIR/main.py" \
        --prompt "$TEXT_PROMPT" \
        --seed "$SEED" \
        --n_samples 1 \
        --W "$MAX_RECT_DIM" \
        --H "$MIN_RECT_DIM" \
        --scales "$SCALE" \
        --outdir "$BASE_DIR/explore/$PROMPT_HASH"
fi
if [[ "$ASPECT_RATIO" == "square" ]];
then
    python "$BASE_DIR/main.py" \
        --prompt "$TEXT_PROMPT" \
        --seed "$SEED" \
        --n_samples 1 \
        --W "$MAX_SQUARE_DIM" \
        --H "$MAX_SQUARE_DIM" \
        --scales "$SCALE" \
        --outdir "$BASE_DIR/explore/$PROMPT_HASH"
fi