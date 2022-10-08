#!/bin/bash -e
# Explore more new images using image from exploration as base
#
# Args: Image file, aspect ratio, scale
# Example:
# ./scripts/continue-exploration.sh [FILENAME] square 9
BASE_DIR="$(dirname "$(dirname "$(realpath $0)")")"
VARY_FILE_NAME="$1"
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

PROMPT_HASH="$(basename "$(dirname "$(realpath $VARY_FILE_NAME)")")"
echo "$PROMPT_HASH"

if [[ ! -d "$BASE_DIR/explore/$PROMPT_HASH" ]];
then
    echo "Could not find folder '$BASE_DIR/explore/$PROMPT_HASH', please use image from explore folders"
    exit 1
fi

TEXT_PROMPT="$(cat $BASE_DIR/explore/$PROMPT_HASH/prompt.txt)"

if [[ -z "$TEXT_PROMPT" ]];
then
    echo "Could not find prompt from '$BASE_DIR/explore/$PROMPT_HASH/prompt.txt', please check it exists or make variations directly with explore tool."
    exit 1
fi

echo "Varying exploration $PROMPT_HASH with $ASPECT_RATIO images using scale $SCALE"
exec "$BASE_DIR/scripts/explore.sh" "$TEXT_PROMPT" "$ASPECT_RATIO" "$SCALE"