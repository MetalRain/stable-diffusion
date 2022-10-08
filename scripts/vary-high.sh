#!/bin/bash -e
# Vary image a lot based on prompt
# bash ./scripts/vary-high.sh [FILENAME] "Nice view"
echo "Varying image $1 for prompt: $2"
prompt_md5=$(bash  ~/repos/stable-diffusion/scripts/start-explore.sh "$2")
python ./img2img.py \
    --init-img "$1" \
    --prompt "$2" \
    --n_samples 1000 \
    --scales 12,11,10,9,9 \
    --strenghts 0.75,0.75,0.7,0.65,0.6 \
    --outdir ./explore/$prompt_md5