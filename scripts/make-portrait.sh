#!/bin/bash -e
# Make single portrait image
# ./scripts/make-portrait.sh "Nice view" 2142523 9 
USER_SCALE=$3
if [[ -z "$USER_SCALE" ]];
then
  USER_SCALE="7"
fi
echo "Generating portrait for prompt: '$1' with seed $2 and scale $USER_SCALE"
prompt_md5=$(bash ~/repos/stable-diffusion/scripts/start-explore.sh "$1")
echo "Images will be in ./explore/$prompt_md5/"
python ./txt2img.py \
    --prompt "$1" \
    --seed "$2" \
    --n_samples 1 \
    --W 448 \
    --H 768 \
    --scales $USER_SCALE \
    --outdir ./explore/$prompt_md5
