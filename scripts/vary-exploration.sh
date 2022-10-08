#!/bin/bash -e
# Vary more images using image from exploration as base
#
# Args: Image file, aspect ratio, vary amount
# Example:
# ./scripts/vary-exploration.sh [FILENAME] square high
BASE_DIR="$(dirname "$(dirname "$(realpath $0)")")"
VARY_FILE_NAME="$1"
ASPECT_RATIO="$2"
if [[ -z "$ASPECT_RATIO" ]];
then
  ASPECT_RATIO="portrait"
fi
VARY_AMOUNT="$3"
if [[ -z "$VARY_AMOUNT" ]];
then
  VARY_AMOUNT="normal"
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

echo "Varying exploration $PROMPT_HASH with $ASPECT_RATIO images using $VARY_AMOUNT of variation"
exec "$BASE_DIR/scripts/vary.sh" "$VARY_FILE_NAME" "$TEXT_PROMPT" "$VARY_AMOUNT"