#!/bin/bash -e
# Explore more new images using image from exploration as base
#
# Args: Image file, aspect ratio, scale
# Example:
# ./scripts/continue-exploration.sh [FILENAME] square 9
BASE_DIR="$(dirname "$(dirname "$(realpath $0)")")"
source "$BASE_DIR/scripts/config.sh"

VARY_FILE_NAME="$1"
FRAMES_IN_SECOND="$2"
if [[ -z "$FRAMES_IN_SECOND" ]];
then
  FRAMES_IN_SECOND="3"
fi

PROMPT_HASH="$(basename "$(dirname "$(realpath $VARY_FILE_NAME)")")"

if [[ ! -d "$BASE_DIR/explore/$PROMPT_HASH" ]];
then
    echo "Could not find folder '$BASE_DIR/explore/$PROMPT_HASH', please use image from explore folders"
    exit 1
fi

echo "Creating video from images in $BASE_DIR/explore/$PROMPT_HASH"
ffmpeg -framerate "$FRAMES_IN_SECOND" -pattern_type glob -i "$BASE_DIR/explore/$PROMPT_HASH/*.png" -r 30 -pix_fmt yuv420p -b:v 6000k "$BASE_DIR/explore/$PROMPT_HASH/video.mp4"
echo "Done!"