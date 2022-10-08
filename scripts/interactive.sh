#!/bin/bash -e
# Interactively make new images from prompts
# Save prompts for further exploration
#
# Args: Prompt, aspect ratio, scale
# Example:
# ./scripts/interactive.sh "Nice view" portrait 9
BASE_DIR="$(dirname "$(dirname "$(realpath $0)")")"
mkdir -p  $BASE_DIR/explore/scratch

TEXT_PROMPT="$1"
ASPECT_RATIO="$2"
if [[ -z "$ASPECT_RATIO" ]];
then
  ASPECT_RATIO="portrait"
fi
SCALE="$3"
if [[ -z "$SCALE" ]];
then
  SCALE="5"
fi

if [[ -z "$TEXT_PROMPT" ]];
then
    echo "Give prompt for next image:"
    read prompt
    if [[ "q" == "$prompt" ]];
    then
        echo "Closing off.."
        exit 0
    fi
else
    prompt="$TEXT_PROMPT"
fi

while [[ ! -z "$prompt" ]];
do 
    bash "$BASE_DIR/scripts/scratch.sh" "$prompt" "$ASPECT_RATIO" "$SCALE"
    echo "Current prompt: '$prompt'"
    echo "How do you want to proceed?"
    echo "Sample again with same prompt [enter], Edit prompt [e], Save prompt & explore further [s], Quit [q]"
    read -e new_prompt
    if [[ ! -z "$new_prompt" ]];
    then
        if [[ "e" == "$new_prompt" ]];
        then
            echo "Edit prompt:"
            read -e -i "$prompt" new_prompt
        fi
        if [[ "q" == "$new_prompt" ]];
        then
            echo "Closing off.."
            break;
        fi
        if [[ "s" == "$new_prompt" ]];
        then
            echo "Saving prompt.."
            PROMPT_HASH="$(bash $BASE_DIR/scripts/init-explore.sh "$prompt")"
            echo "Prompt saved, starting exploration.."
            exec "$BASE_DIR/explore/$PROMPT_HASH/explore.sh" "$ASPECT_RATIO" "7"
        fi
        prompt="$new_prompt"
    fi
done