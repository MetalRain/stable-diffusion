#!/bin/bash -e
# Vary image "normal" amount one based on prompt
# bash ./scripts/vary.sh [FILENAME] "Nice view"
echo "Varying image $1 for prompt: $2"
prompt_md5=$(bash  ~/repos/stable-diffusion/scripts/start-explore.sh "$2")
python ./img2img.py \
    --init-img "$1" \
    --prompt "$2" \
    --n_samples 4 \
    --scales 6,7,8,9 \
    --strenghts 0.4,0.45,0.5,0.55 \
    --outdir ./explore/$prompt_md5
