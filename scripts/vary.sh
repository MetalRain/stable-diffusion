#!/bin/bash -e
# Vary image "normal" amount one based on prompt
# bash ./scripts/vary.sh [FILENAME] "Nice view" high
VARY_AMOUNT=$3
if [[ -z "$VARY_AMOUNT" ]];
then
  VARY_AMOUNT="normal"
fi
echo "Varying image $1 for prompt: '$2' using $VARY_AMOUNT of variation"
prompt_md5=$(bash  ~/repos/stable-diffusion/scripts/start-explore.sh "$2")
echo "Images will be in ./explore/$prompt_md5/"
if [[ "$VARY_AMOUNT" == "normal" ]];
then
    python ./img2img.py \
        --init-img "$1" \
        --prompt "$2" \
        --n_samples 1000 \
        --scales 6,7,8,9 \
        --strenghts 0.4,0.45,0.5,0.55 \
        --outdir ./explore/$prompt_md5
fi
if [[ "$VARY_AMOUNT" == "low" ]];
then
    python ./img2img.py \
        --init-img "$1" \
        --prompt "$2" \
        --n_samples 1000 \
        --scales 3,3.5,4,4.5 \
        --strenghts 0.3,0.35,0.4,0.45 \
        --outdir ./explore/$prompt_md5
fi
if [[ "$VARY_AMOUNT" == "little" ]];
then
    python ./img2img.py \
        --init-img "$1" \
        --prompt "$2" \
        --n_samples 1000 \
        --scales 10,5,9,6,8,7 \
        --strenghts 0.3,0.35,0.4,0.45,0.5,0.55 \
        --outdir ./explore/$prompt_md5
fi
if [[ "$VARY_AMOUNT" == "high" ]];
then
    python ./img2img.py \
        --init-img "$1" \
        --prompt "$2" \
        --n_samples 1000 \
        --scales 12,11,10,9,9 \
        --strenghts 0.75,0.75,0.7,0.65,0.6 \
        --outdir ./explore/$prompt_md5
fi
if [[ "$VARY_AMOUNT" == "test" ]];
then
    python ./img2img.py \
        --init-img "$1" \
        --prompt "$2" \
        --n_samples 1000 \
        --scales 5 \
        --outdir ./explore/$prompt_md5
fi