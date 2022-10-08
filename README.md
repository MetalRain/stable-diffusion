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

5. Generate images
Interactively
```sh
bash ./scripts/interactive.sh
```

