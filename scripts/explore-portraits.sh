#!/bin/bash -e
# Continously create portrait images from single prompt
# exploring the options 
# ./scripts/make-portrait.sh "Nice view" 9 
USER_SCALE=$2
if [[ -z "$USER_SCALE" ]];
then
  USER_SCALE="5"
fi
echo "Exploring portraits for prompt: '$1' using scale: ${USER_SCALE}"
prompt_md5=$(bash  ~/repos/stable-diffusion/scripts/start-explore.sh "$1")
echo "Images will be in ./explore/$prompt_md5/"
python ./txt2img.py \
    --prompt "$1" \
    --n_samples 1000 \
    --W 448 \
    --H 768 \
    --scales $USER_SCALE \
    --outdir ./explore/$prompt_md5