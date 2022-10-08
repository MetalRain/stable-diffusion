#!/bin/bash -e
# Vary image a little based on prompt
# bash ./scripts/vary-little.sh [FILENAME] "Nice view"
echo "Varying image $1 for prompt: $2"
prompt_md5=$(bash  ~/repos/stable-diffusion/scripts/start-explore.sh "$2")
python ./img2img.py \
    --init-img "$1" \
    --prompt "$2" \
    --n_samples 1000 \
    --scales 10,5,9,6,8,7 \
    --strenghts 0.3,0.35,0.4,0.45,0.5,0.55 \
    --outdir ./explore/$prompt_md5