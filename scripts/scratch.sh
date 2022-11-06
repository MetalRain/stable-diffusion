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

echo "Exploring $ASPECT_RATIO images for prompt: '$TEXT_PROMPT' using scale $SCALE"
echo "You can find images in $BASE_DIR/explore/scratch"

IMAGE_W="$MIN_RECT_DIM"
IMAGE_H="$MAX_RECT_DIM"
if [[ "$ASPECT_RATIO" == "portrait" ]];
then
  IMAGE_W="$MIN_RECT_DIM"
  IMAGE_H="$MAX_RECT_DIM"
fi
if [[ "$ASPECT_RATIO" == "landscape" ]];
then
  IMAGE_W="$MAX_RECT_DIM"
  IMAGE_H="$MIN_RECT_DIM"
fi
if [[ "$ASPECT_RATIO" == "square" ]];
then
  IMAGE_W="$MAX_SQUARE_DIM"
  IMAGE_H="$MAX_SQUARE_DIM"
fi
if [[ "$ASPECT_RATIO" == "optimal" ]];
then
  IMAGE_W="512"
  IMAGE_H="512"
fi

python "$BASE_DIR/main.py" \
    --prompt "$TEXT_PROMPT" \
    --n_samples "$INTERACTIVE_IMAGES" \
    --W "$IMAGE_W" \
    --H "$IMAGE_H" \
    --scales "$SCALE" \
    --task "txt2img" \
    --outdir "$BASE_DIR/explore/scratch"