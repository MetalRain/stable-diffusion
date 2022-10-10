#!/bin/bash -e
# Vary image based on prompt
#
# Args: Image file, prompt, vary amount, images to generate
# Example:
# ./scripts/vary.sh [FILENAME] "Nice view" high 20
BASE_DIR="$(dirname "$(dirname "$(realpath $0)")")"
VARY_FILE_NAME="$1"
TEXT_PROMPT="$2"
VARY_AMOUNT="$3"
if [[ -z "$VARY_AMOUNT" ]];
then
  VARY_AMOUNT="normal"
fi
VARY_SAMPLES="$4"
if [[ -z "$VARY_SAMPLES" ]];
then
  VARY_SAMPLES="3"
fi

echo "Varying image '$VARY_FILE_NAME' for prompt: '$TEXT_PROMPT' using '$VARY_AMOUNT' transformation"
PROMPT_HASH=$(bash $BASE_DIR/scripts/init-explore.sh "$TEXT_PROMPT")
echo "Images will be in $BASE_DIR/explore/$PROMPT_HASH/"
if [[ "$VARY_AMOUNT" == "normal" ]];
then
    python "$BASE_DIR/img2img.py" \
        --init-img "$1" \
        --prompt "$2" \
        --n_samples "$VARY_SAMPLES" \
        --scales "6,7,8,9" \
        --strenghts "0.35,0.4,0.45,0.5" \
        --outdir "$BASE_DIR/explore/$PROMPT_HASH" \
        --waits "0,0,0"
fi
if [[ "$VARY_AMOUNT" == "mild" ]];
then
    python "$BASE_DIR/img2img.py" \
        --init-img "$1" \
        --prompt "$2" \
        --n_samples "$VARY_SAMPLES" \
        --scales "6,7,8,9" \
        --strenghts "0.2,0.25,0.3,0.35" \
        --outdir "$BASE_DIR/explore/$PROMPT_HASH" \
        --waits "0,0,0"
fi
if [[ "$VARY_AMOUNT" == "unfocus" ]];
then
    python "$BASE_DIR/img2img.py" \
        --init-img "$1" \
        --prompt "$2" \
        --n_samples "$VARY_SAMPLES" \
        --scales "3,3.5,4" \
        --strenghts "0.8,0.85,0.9" \
        --outdir "$BASE_DIR/explore/$PROMPT_HASH" \
        --waits "0,0,0"
fi
if [[ "$VARY_AMOUNT" == "fixit" ]];
then
    python "$BASE_DIR/img2img2img.py" \
        --init-img "$1" \
        --prompt "$2" \
        --n_samples "$VARY_SAMPLES" \
        --scales "4,6,9,11" \
        --strenghts "0.8,0.5,0.3,0.15" \
        --outdir "$BASE_DIR/explore/$PROMPT_HASH" \
        --waits "0,0,0"
fi
if [[ "$VARY_AMOUNT" == "restyle" ]];
then
    python "$BASE_DIR/img2img2img.py" \
        --init-img "$1" \
        --prompt "$2" \
        --n_samples "$VARY_SAMPLES" \
        --scales "5.5,7.2,9.8" \
        --strenghts "0.45,0.4,0.35" \
        --outdir "$BASE_DIR/explore/$PROMPT_HASH" \
        --waits "0,0,0"
fi
if [[ "$VARY_AMOUNT" == "retouch" ]];
then
    python "$BASE_DIR/img2img2img.py" \
        --init-img "$1" \
        --prompt "$2" \
        --n_samples "$VARY_SAMPLES" \
        --scales "9,10,10,11" \
        --strenghts "0.2,0.2,0.2,0.15" \
        --outdir "$BASE_DIR/explore/$PROMPT_HASH" \
        --waits "0,0,0"
fi
if [[ "$VARY_AMOUNT" == "shake" ]];
then
    python "$BASE_DIR/img2img.py" \
        --init-img "$1" \
        --prompt "$2" \
        --n_samples "$VARY_SAMPLES" \
        --scales "10,5,9,6,8,7" \
        --strenghts "0.3,0.35,0.4,0.45,0.5,0.55" \
        --outdir "$BASE_DIR/explore/$PROMPT_HASH" \
        --waits "0,0,0"
fi
if [[ "$VARY_AMOUNT" == "unstuck" ]];
then
    python "$BASE_DIR/img2img.py" \
        --init-img "$1" \
        --prompt "$2" \
        --n_samples "$VARY_SAMPLES" \
        --scales "5,6,7" \
        --strenghts "0.8,0.8,0.8" \
        --outdir "$BASE_DIR/explore/$PROMPT_HASH" \
        --waits "0,0,0"
fi
if [[ "$VARY_AMOUNT" == "focus" ]];
then
    python "$BASE_DIR/img2img.py" \
        --init-img "$1" \
        --prompt "$2" \
        --n_samples "$VARY_SAMPLES" \
        --scales "12,11,10,9,9" \
        --strenghts "0.75,0.75,0.7,0.65,0.6" \
        --outdir "$BASE_DIR/explore/$PROMPT_HASH" \
        --waits "0,0,0"
fi