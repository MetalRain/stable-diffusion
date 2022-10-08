# Stable diffusion batch CLI

CLI tools for cataloguing prompt results and creating variations in large quantities.

Modified versions

More detailed install instructions:
https://www.assemblyai.com/blog/how-to-run-stable-diffusion-locally-to-generate-images/

## Install instructions

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
You should have `sd-v1-4.ckpt` file at root of this folder.

## Usage

### Explore prompts images
Start from interactive CLI tool
```sh
bash ./scripts/interactive.sh "Art piece" square
bash ./scripts/interactive.sh "Nice view" landscape
bash ./scripts/interactive.sh "Starry sky" portrait
```

Once you have something worthwhile, save prompt and tool starts exploring.

Later you can also explore prompt with varying scales and aspect ratios
```sh
bash ./explore/[prompt-hash]/explore.sh portrait 7
bash ./explore/[prompt-hash]/explore.sh square 9
```

And make variations from existing images
```sh
bash ./explore/[prompt-hash]/vary.sh ./explore/[prompt-hash]/[image-file].png "" normal
```

Higher variation profile generates more extreme versions of same prompt
```sh
bash ./explore/[prompt-hash]/vary.sh ./explore/[prompt-hash]/[image-file].png "" high
```

You can also create new prompt by enhancing existing prompts
```sh
cd ./explore/[prompt-hash]
bash ./explore/[prompt-hash]/vary.sh ./explore/[prompt-hash]/[image-file].png ", words to be added to prompt" normal
```
