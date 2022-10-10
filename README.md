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
bash ./scripts/interactive.sh "Nice view" landscape 7
bash ./scripts/interactive.sh "Starry sky" portrait 9
```

Once you have found something worthwhile, save the prompt and tool starts exploring.

Later you can also explore prompt with varying scales and aspect ratios
to find more interesting images
```sh
bash ./scripts/continue-exploration.sh ./explore/[prompt-hash]/prompt.txt portrait 7
bash ./scripts/continue-exploration.sh ./explore/[prompt-hash]/prompt.txt square 10
```

And make similar images from existing images, transformations available:
* normal: some mixture of focusing and unfocucsing, useful when you don't really know how to proceed
* shake: tries high scales with low strenghs and vice versa, use when image is good but bit bland
* fixit: apply lowering and then focus in, really makes your image as commanded
* restyle: apply lowering and then focus, similar to fixit, but less aggresive
* focus: enhance things already in picture, useful when honing in to specific style, may backfire
* unstuck: changes to structure or style, may lose some desired qualities from image
* unfocus: totally lose focus and think big, good for discovery and then honing back in
Don't make too many consecutive transforms on image or it will be cornered.
Try to first make good image with trying different prompts!

```sh
bash ./scripts/vary-exploration.sh ./explore/[prompt-hash]/[filename].png normal
bash ./scripts/vary-exploration.sh ./explore/[prompt-hash]/[filename].png change
bash ./scripts/vary-exploration.sh ./explore/[prompt-hash]/[filename].png refine
```

You can also create new prompt by enhancing existing prompts
```sh
bash ./scripts/vary-exploration.sh ./explore/[prompt-hash]/[filename].png normal ", in space" 
```
