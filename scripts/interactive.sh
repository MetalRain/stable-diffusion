#!/bin/bash -e
# Interactively make new images from prompts
# Save prompts for further exploration
#
# Args: Prompt, aspect ratio, scale
# Example:
# ./scripts/interactive.sh "Nice view" portrait 9
BASE_DIR="$(dirname "$(dirname "$(realpath $0)")")"
source "$BASE_DIR/scripts/config.sh"

mkdir -p  $BASE_DIR/explore/scratch

TEXT_PROMPT="$1"
ASPECT_RATIO="$2"
if [[ -z "$ASPECT_RATIO" ]];
then
  ASPECT_RATIO="$DEFAULT_ASPECT_RATIO"
fi
SCALE="$3"
if [[ -z "$SCALE" ]];
then
  SCALE="$DEFAULT_SCALE"
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
    echo "---"
    echo "Current prompt: '$prompt'"
    echo "Aspect ratio: $ASPECT_RATIO, scale: $SCALE"
    echo "---"
    echo "How do you want to proceed?, here are our options:"
    echo "Generate new images [g], Edit prompt [e]"
    echo "Adjust scale [c], Change aspect ratio [a]"
    echo "Save & explore further [s], Quit [q]"
    read -e command_prompt
    if [[ ! -z "$command_prompt" ]];
    then
        if [[ "e" == "$command_prompt" ]];
        then
            echo "Edit prompt:"
            new_prompt=""
            read -e -i "$prompt" new_prompt
            prompt="$new_prompt"
        fi
        if [[ "g" == "$command_prompt" ]];
        then
            echo "Generating new images.."
            bash "$BASE_DIR/scripts/scratch.sh" "$prompt" "$ASPECT_RATIO" "$SCALE"
        fi
        if [[ "c" == "$command_prompt" ]];
        then
            echo "Give new scale, numeric value between 1-30, use . as decimal separator:"
            read -e SCALE
        fi
        if [[ "a" == "$command_prompt" ]];
        then
            echo "Give new aspect ratio, either square, portrait or landscape:"
            read -e ASPECT_RATIO
        fi
        if [[ "q" == "$command_prompt" ]];
        then
            echo "Closing off.."
            break;
        fi
        if [[ "s" == "$command_prompt" ]];
        then
            echo "Saving prompt.."
            PROMPT_HASH="$(bash $BASE_DIR/scripts/init-explore.sh "$prompt")"
            echo "Prompt saved, starting exploration.."
            bash "$BASE_DIR/scripts/explore.sh" "$prompt" "$ASPECT_RATIO" "$SCALE"
        fi
    fi
done