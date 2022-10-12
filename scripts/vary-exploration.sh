#!/bin/bash -e
# Vary more images using image from exploration as base
#
# Args: Image file, vary amount, prompt_addition
# Example:
# ./scripts/vary-exploration.sh [FILENAME] high ", natural sunlight"
BASE_DIR="$(dirname "$(dirname "$(realpath $0)")")"
VARY_FILE_NAME="$1"
VARY_AMOUNT="$2"
if [[ -z "$VARY_AMOUNT" ]];
then
  VARY_AMOUNT="normal"
fi
PROMPT_ADDITION="$3"
if [[ -z "$PROMPT_ADDITION" ]];
then
  PROMPT_ADDITION=""
fi
VARY_SAMPLES="$4"
if [[ -z "$VARY_SAMPLES" ]];
then
  VARY_SAMPLES="3"
fi

PROMPT_HASH="$(basename "$(dirname "$(realpath $VARY_FILE_NAME)")")"
echo "$PROMPT_HASH"

if [[ ! -d "$BASE_DIR/explore/$PROMPT_HASH" ]];
then
    echo "Could not find folder '$BASE_DIR/explore/$PROMPT_HASH', please use image from explore folders"
    exit 1
fi

TEXT_PROMPT="$(cat $BASE_DIR/explore/$PROMPT_HASH/prompt.txt)$PROMPT_ADDITION"

if [[ -z "$TEXT_PROMPT" ]];
then
    echo "Could not find prompt from '$BASE_DIR/explore/$PROMPT_HASH/prompt.txt', please check it exists or make variations directly with explore tool."
    exit 1
fi

echo "Varying exploration $PROMPT_HASH using $VARY_AMOUNT variation"
exec "$BASE_DIR/scripts/vary.sh" "$VARY_FILE_NAME" "$TEXT_PROMPT" "$VARY_AMOUNT" "$VARY_SAMPLES"