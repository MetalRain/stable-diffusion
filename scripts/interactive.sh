#!/bin/bash -e
# Interactively make new images from prompts
# Save prompts for further exploration
function cleanup {
  echo "Cleaning up.."
  cd ~/repos/stable-diffusion
}
trap cleanup EXIT
mkdir -p  ~/repos/stable-diffusion/explore/scratch

USER_SCALE=$1
if [[ -z "$USER_SCALE" ]];
then
  USER_SCALE="5"
fi

echo "Give prompt for next image:"
read prompt
if [[ "q" == "$prompt" ]];
then
    echo "Closing off.."
    exit 0
fi
while [[ ! -z "$prompt" ]];
do 
    echo "Exploring portraits for prompt: '$prompt'"
    python ./txt2img.py \
        --prompt "$prompt" \
        --ckpt sd-v1-4.ckpt \
        --n_samples 3 \
        --W 448 \
        --H 768 \
        --scales "$USER_SCALE" \
        --outdir ./explore/scratch
    echo "Current prompt: '$prompt'"
    echo "Continue with same [enter], Edit prompt [e], Save prompt & exit [s], Quit [q]"
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
            prompt_md5=$(bash ~/repos/stable-diffusion/scripts/start-explore.sh "$prompt")
            echo "Prompt saved, starting exploration.."
            cd "./explore/$prompt_md5"
            exec bash "./explore/$prompt_md5/explore.sh" 7
        fi
        prompt="$new_prompt"
    fi
done