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
  SCALE="6"
fi

if [[ -z "$TEXT_PROMPT" ]];
then
    echo "Give prompt for next image:"
    read -e prompt
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
    echo "How do you want to proceed?, here are our options:"
    printf "Sample new images with same prompt [enter], Edit prompt [e]\nAdjust scale [c], Change aspect ratio [a]\nSave & explore further [s], Quit [q]\n"
    read -e new_prompt
    if [[ ! -z "$new_prompt" ]];
    then
        if [[ "e" == "$new_prompt" ]];
        then
            echo "Edit prompt:"
            read -e -i "$prompt" new_prompt
        fi
        if [[ "c" == "$new_prompt" ]];
        then
            echo "Give new scale, numeric value between 1-30, use . as decimal separator:"
            read -e SCALE
            new_prompt="$prompt"
        fi
        if [[ "a" == "$new_prompt" ]];
        then
            echo "Give new aspect ratio, either square, portrait or landscape:"
            read -e ASPECT_RATIO
            new_prompt="$prompt"
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
            exec "$BASE_DIR/scripts/explore.sh" "$prompt" "$ASPECT_RATIO" "$SCALE"
        fi
        prompt="$new_prompt"
    fi
done