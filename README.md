# Stable diffusion batch CLI

CLI tools for cataloguing prompt results and creating variations in large quantities.

This configuration requires beefy GPU, like RTX 2080Ti beefy.

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

Interactive cli saves images to `./explore/scratch` folder.

Once you have found something worthwhile, save the prompt and tool starts exploring.

Later you can also explore prompt with varying scales and aspect ratios
to find more interesting images
```sh
bash ./scripts/continue-exploration.sh ./explore/[prompt-hash]/prompt.txt portrait 7
bash ./scripts/continue-exploration.sh ./explore/[prompt-hash]/prompt.txt square 10
```

Main point of these scripts is this exploration, you can find good prompts and then leave your computer cranking more images.

Images are saved with unique names that will have some metadata on them:
* Creation timestamp
* Scale
* Strength
* PLMS/DDIM steps
* Used seed


### Vary prompts images

Make similar images from existing images and drive results towards desired state.

text2img and img2img basically have two major parameters: scale and strenght.
* Scale affects how detailed & noisy image will be. Scales can be set in about 1 - 30 range. Higher scales take more time to process so
* Strength affects how much of the image will be changed. Strength is value between 0.0 - 1.0

There are few transformations available:

Single step transforms:
* normal: mixture of low strength transforms, mostly 6-7 scale, but some 10s to kick to ball forward
* shake: tries high scales with low strenghs and vice versa, produces lot of similarish images that you can more easilty choose some next.
* focus: try to force prompt to happen by widely changing image using high strengths and scales.
* unstuck: changes to structure or style, may lose some desired qualities from image, uses high strengths and low scales.
* unfocus: totally lose focus and think big, good for discovery and then honing back in. Uses ultra low scales and high strengths.

Multi step transforms:
* beast: series of low strength transforms, 6-6-6 scales, compromises control over throughput 
* god: kind of like beast, series of three low strength transforms, but scale goes up. Idea is that beast does the work but may end up ugly, but god always ends up good even if less work is being done.
* enhance: longer series of transforms, tries to first clean up image lower scale then use higher scales to bring quality up. Betten than god for blurry images.
* finalize: adds some little high scale touches to image, may be useful after long refinement in lower scales

Multi step transforms take image from img2img and pass it multiple times with different values.

Ideally you get serviceable image from prompt, if it's far of target then use `focus` or `god`, if structure is correct but there are many wrong things in image, try `shake` or `beast`. If you think you are close you can use normal to advance more slowly, until you go too far and have to stop or try earlier image.

Don't make too many consecutive small transforms on image or it will start to deteriorate in image quality. Sometimes model "recognizes noice as the pattern" and starts to amplify it. Having better starting image by experimenting more with prompts will pay off. Sometimes it's worthwhile to run high scale transform like `focus` or `unstuck` to "renew" image quality before refining further.

```sh
bash ./scripts/vary-exploration.sh ./explore/[prompt-hash]/[filename].png normal
bash ./scripts/vary-exploration.sh ./explore/[prompt-hash]/[filename].png enhance
bash ./scripts/vary-exploration.sh ./explore/[prompt-hash]/[filename].png unstuck
```

You can also create & save new prompt by adding more words to existing prompts
```sh
bash ./scripts/vary-exploration.sh ./explore/[prompt-hash]/[filename].png normal ", in space" 
```

### Using ready images

Just put your images as PNG files, their dimensions should be some multiple of 64.

In my case my GPU can handle:
* 448x768 (or 768x448)
* 512x512
If you encounter memory errors, try to lower these values in `./scripts/config.sh`

Then you can start with: 
```sh
bash ./scripts/vary.sh [path-to-my-image].png "Lovely weather, painting" shake
```

This creates exploration folder in `./explore/[prompt-hash]` like you would have started with interactive mode.

Since image already has something you want to preserve, use some low strength transform like `shake` to get many files where you can start your exploration. Also `normal` work just fine as always.


## Configuration

Script default values are configured from `./scripts/config.sh`.

You can adjust:
* Image sizes
* Defaults
* Wait times

## Image correction

Stable diffusion model seems to skew results towards magenta.
This is especially unconvinient when you want to repeatedly transform image.

Corrections these scripts do:
* Convert image to CIELAB
* Color correct A* and B* channels based on input image histogram
* Boost L* channel to avoid darkening image

L* channel is not adjusted based on histogram since that causes visible color banding.
This still doesn't remove all magenta hue, there are some blended blotches of that in output images and also fringing artefacts from color correction.

However in general these have helped a lot.

Things still to do:
* Resample and smooth color correction target in long sequences (now just input image)
* color correct after every transform (now just on save)
* Adjust color correction & boost intensity based on transform parameters
