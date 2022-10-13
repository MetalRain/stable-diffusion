#!/bin/bash -e
# Vary image based on prompt
#
# Args: Image file, prompt, vary amount, images to generate
# Example:
# ./scripts/vary.sh [FILENAME] "Nice view" high 20
BASE_DIR="$(dirname "$(dirname "$(realpath $0)")")"
source "$BASE_DIR/scripts/config.sh"

VARY_FILE_NAME="$1"
TEXT_PROMPT="$2"
VARY_AMOUNT="$3"
if [[ -z "$VARY_AMOUNT" ]];
then
  VARY_AMOUNT="$DEFAULT_VARIATION_TRANSFORM"
fi
VARY_SAMPLES="$4"
if [[ -z "$VARY_SAMPLES" ]];
then
  VARY_SAMPLES="$DEFAULT_VARIATION_ITERATIONS"
fi

echo "Varying image '$VARY_FILE_NAME' for prompt: '$TEXT_PROMPT' using '$VARY_AMOUNT' transformation"
PROMPT_HASH=$(bash $BASE_DIR/scripts/init-explore.sh "$TEXT_PROMPT")
echo "Images will be in $BASE_DIR/explore/$PROMPT_HASH/"

if [[ "$VARY_AMOUNT" == "normal" ]];
then
    SINGLE_PER_LOOP=""
    SCALES="6,7,10"
    STRENGHTS="0.35,0.45,0.3"

fi
if [[ "$VARY_AMOUNT" == "unstuck" ]];
then
    SINGLE_PER_LOOP=""
    SCALES="5,6,7"
    STRENGHTS="0.8,0.8,0.8"

fi
if [[ "$VARY_AMOUNT" == "unfocus" ]];
then
    SINGLE_PER_LOOP=""
    SCALES="3,3.5,4"
    STRENGHTS="0.8,0.85,0.9"

fi
if [[ "$VARY_AMOUNT" == "focus" ]];
then
    SINGLE_PER_LOOP=""
    SCALES="12,10,8"
    STRENGHTS="0.7,0.7,0.7"

fi
if [[ "$VARY_AMOUNT" == "shake" ]];
then
    SINGLE_PER_LOOP=""
    SCALES="10,5,9,6,8,7"
    STRENGHTS="0.3,0.35,0.4,0.45,0.5,0.55"
fi

if [[ "$VARY_AMOUNT" == "beast" ]];
then
    SINGLE_PER_LOOP="1"
    SCALES="6,6,6"
    STRENGHTS="0.35,0.35,0.35"
fi
if [[ "$VARY_AMOUNT" == "god" ]];
then
    SINGLE_PER_LOOP="1"
    SCALES="6,9,11"
    STRENGHTS="0.35,0.35,0.35"
fi
if [[ "$VARY_AMOUNT" == "finalize" ]];
then
    SINGLE_PER_LOOP="1"
    SCALES="8,11,15"
    STRENGHTS="0.3,0.2,0.1"
fi
if [[ "$VARY_AMOUNT" == "enhance" ]];
then
    SINGLE_PER_LOOP="1"
    SCALES="6,6,7,7,9"
    STRENGHTS="0.2,0.2,0.3,0.5,0.2"
fi


if [[ -z "$SINGLE_PER_LOOP" ]];
then
    python "$BASE_DIR/img2img2img.py" \
        --init-img "$1" \
        --prompt "$2" \
        --n_samples "$VARY_SAMPLES" \
        --scales "$SCALES" \
        --strenghts "$STRENGHTS" \
        --outdir "$BASE_DIR/explore/$PROMPT_HASH" \
        --waits "$VARIATION_WAITS"
else
    python "$BASE_DIR/img2img2img.py" \
        --init-img "$1" \
        --prompt "$2" \
        --n_samples "$VARY_SAMPLES" \
        --scales "$SCALES" \
        --strenghts "$STRENGHTS" \
        --outdir "$BASE_DIR/explore/$PROMPT_HASH" \
        --waits "$VARIATION_WAITS" \
        --image_per_loop
fi
