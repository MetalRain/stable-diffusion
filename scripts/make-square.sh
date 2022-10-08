#!/bin/bash -e
# Make single square image
# ./scripts/make-square.sh "Nice view" 2142523 9 
USER_SCALE=$3
if [[ -z "$USER_SCALE" ]];
then
  USER_SCALE="7"
fi
echo "Generating image for prompt: $1"
prompt_md5=$(bash ~/repos/stable-diffusion/scripts/start-explore.sh "$1")
echo "Images will be in ./explore/$prompt_md5/"
python ./txt2img.py \
    --prompt "$1" \
    --n_samples 1 \
    --scales $USER_SCALE \
    --W 512 \
    --H 512 \
    --outdir ./explore/$prompt_md5
