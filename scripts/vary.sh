#!/bin/bash -e
# Vary image based on prompt
#
# Args: Image file, prompt, vary amount
# Example:
# ./scripts/vary.sh [FILENAME] "Nice view" high
BASE_DIR="$(dirname "$(dirname "$(realpath $0)")")"
VARY_FILE_NAME="$1"
TEXT_PROMPT="$2"
VARY_AMOUNT="$3"
if [[ -z "$VARY_AMOUNT" ]];
then
  VARY_AMOUNT="normal"
fi
echo "Varying image $VARY_FILE_NAME for prompt: '$TEXT_PROMPT' using $VARY_AMOUNT of variation"
PROMPT_HASH=$(bash $BASE_DIR/scripts/init-explore.sh "$TEXT_PROMPT")
echo "Images will be in $BASE_DIR/explore/$PROMPT_HASH/"
if [[ "$VARY_AMOUNT" == "normal" ]];
then
    python "$BASE_DIR/img2img.py" \
        --init-img "$1" \
        --prompt "$2" \
        --n_samples 1000 \
        --scales 6,7,8,9 \
        --strenghts 0.4,0.45,0.5,0.55 \
        --outdir "$BASE_DIR/explore/$PROMPT_HASH"
fi
if [[ "$VARY_AMOUNT" == "low" ]];
then
    python "$BASE_DIR/img2img.py" \
        --init-img "$1" \
        --prompt "$2" \
        --n_samples 1000 \
        --scales 3,3.5,4,4.5 \
        --strenghts 0.3,0.35,0.4,0.45 \
        --outdir "$BASE_DIR/explore/$PROMPT_HASH"
fi
if [[ "$VARY_AMOUNT" == "little" ]];
then
    python "$BASE_DIR/img2img.py" \
        --init-img "$1" \
        --prompt "$2" \
        --n_samples 1000 \
        --scales 10,5,9,6,8,7 \
        --strenghts 0.3,0.35,0.4,0.45,0.5,0.55 \
        --outdir "$BASE_DIR/explore/$PROMPT_HASH"
fi
if [[ "$VARY_AMOUNT" == "high" ]];
then
    python "$BASE_DIR/img2img.py" \
        --init-img "$1" \
        --prompt "$2" \
        --n_samples 1000 \
        --scales 12,11,10,9,9 \
        --strenghts 0.75,0.75,0.7,0.65,0.6 \
        --outdir "$BASE_DIR/explore/$PROMPT_HASH"
fi
if [[ "$VARY_AMOUNT" == "test" ]];
then
    python "$BASE_DIR/img2img.py" \
        --init-img "$1" \
        --prompt "$2" \
        --n_samples 1000 \
        --scales 5 \
        --outdir "$BASE_DIR/explore/$PROMPT_HASH"
fi