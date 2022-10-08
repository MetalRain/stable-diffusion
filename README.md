# Stable diffusion

Full install instructions:
https://www.assemblyai.com/blog/how-to-run-stable-diffusion-locally-to-generate-images/

## Instructions

1. Install conda
```sh
bash ./scripts/Miniconda3-py38_4.12.0-Linux-x86_64.sh
```

2. Clone repo
```sh
git clone https://github.com/CompVis/stable-diffusion.git
cd stable-diffusion/
```

3. Create conda environment
```sh
conda env create -f environment.yaml
conda activate ldm
```

4. Download stable diffusion weights
```sh
curl https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media > sd-v1-4.ckpt
```

5. Explore prompts images
Interactively
```sh
bash ./scripts/interactive.sh
```

6. Explore saved prompts
Explore prompt with varying scales
```sh
bash ./explore/[prompt-hash]/explore.sh 7
bash ./explore/[prompt-hash]/explore.sh 9
```

7. Vary generated images

Some variations with same prompt
```sh
cd ./explore/[prompt-hash]
bash ./vary.sh ./explore/[prompt-hash]/[file-to-vary] "" normal
```

More variations with same prompt
```sh
cd ./explore/[prompt-hash]
bash ./vary.sh ./[file-to-vary] "" high
```

Create another exploration with new prompt
```sh
cd ./explore/[prompt-hash]
bash ./vary.sh ./[file-to-vary] ", words to be added to prompt" normal
```
