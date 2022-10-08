#!/bin/bash -e
# Vary image into more general one based on prompt
# bash ./scripts/vary-low.sh [FILENAME] "Nice view"
echo "Varying image $1 for prompt: $2"
prompt_md5=$(bash  ~/repos/stable-diffusion/scripts/start-explore.sh "$2")
python ./img2img.py \
    --init-img "$1" \
    --prompt "$2" \
    --n_samples 1000 \
    --scales 3,3.5,4,4.5 \
    --strenghts 0.3,0.35,0.4,0.45 \
    --outdir ./explore/$prompt_md5