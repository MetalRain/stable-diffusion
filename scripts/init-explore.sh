#!/bin/bash
# Save prompt and create exploration scripts
# Creates folder ./explore
# and inside of that new folder for each prompt
#
# Args: Prompt
# Example:
# ./scripts/init-explores.sh "Nice view"
TEXT_PROMPT="$1"
PROMPT_HASH=`/bin/echo $TEXT_PROMPT | /usr/bin/md5sum | /bin/cut -f1 -d" "`
BASE_DIR="$(dirname "$(dirname "$(realpath $0)")")"
mkdir -p $BASE_DIR/explore/$PROMPT_HASH
echo "$1" > $BASE_DIR/explore/$PROMPT_HASH/prompt.txt
echo "$PROMPT_HASH"