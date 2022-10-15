"""make variations of input image"""
from importlib import import_module
import datetime, time
import random
import argparse, os
import PIL
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange, repeat
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything

# from ldm.util import instantiate_from_config
ldm_util = import_module("stable-diffusion.ldm.util")
instantiate_from_config = ldm_util.instantiate_from_config

# from ldm.models.diffusion.ddim import DDIMSampler
ddim = import_module("stable-diffusion.ldm.models.diffusion.ddim")
DDIMSampler = ddim.DDIMSampler


# Color correction from skimage
# https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/exposure/histogram_matching.py#L24-L85

def cdf_prep(template, boost=0.0):
    """Prepare template values for cdf calculation"""
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)
    # calculate normalized quantiles for each array
    tmpl_quantiles = np.cumsum(tmpl_counts) / float(template.size)
    # move quantiles down to match darkening
    tmpl_quantiles = np.clip(tmpl_quantiles + boost, 0.0, 1.0)
    return (tmpl_quantiles, tmpl_values)

def match_cdf(source, ref_quantiles, ref_values, blend_amount=0.5):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    src_values, src_unique_indices, src_counts = np.unique(
        source.ravel(),
        return_inverse=True,
        return_counts=True
    )

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / float(source.size)

    # interpolate values for each unique index
    interp_a_values = np.interp(src_quantiles, ref_quantiles, ref_values)
    new_values = interp_a_values[src_unique_indices]

    # Blend source and interpolation to avoid too much error
    inv_blend = 1.0 - blend_amount
    return (
        source.ravel() * blend_amount
        + new_values * inv_blend
    ).reshape(source.shape)


def color_correct_lab_image(lab_image, lab_correction):
    """
    Match color channels of CIELAB color space image to reference image
    
    Very similar to:
    skimage.exposure.match_histograms
    ---
    Adjust an image so that its cumulative histogram matches that of another.
    The adjustment is applied separately for each channel.
    ---
    .. [1] http://paulbourke.net/miscellaneous/equalisation/
    """
    # Blend L channel less times, since it's more susceptible to
    # make image color band, keep 0.9^2 = 81% of incoming L
    # In order to combat against darkening, adjust L channel quantiles
    # Color channels A and B get smoothed over multiple times
    # only preserving 0.4^4 = 2.6% original color and rest coming from reference
    # this keeps drift to magenta to minimum whilst allowing colors to change
    channel_quantile_boost = [-0.08, 0.00, 0.0]
    channel_blend_iterations = [2, 4, 4]
    channel_blend_amount=[0.9, 0.4, 0.4]
    matched = np.empty(lab_image.shape, dtype=lab_image.dtype)
    
    # Adjust channels
    for channel in range(lab_image.shape[-1]):
        source_channel = lab_image[..., channel]
        reference_channel = lab_correction[..., channel]
        # Use same ref for all rounds
        
        ref_quantiles, ref_values = cdf_prep(
            template=reference_channel,
            boost=channel_quantile_boost[channel]
        )
        matched_channel = source_channel
        # Blend several times to soften otherwise hard changes
        for _ in range(channel_blend_iterations[channel]):
            matched_channel = match_cdf(
                source=matched_channel,
                ref_quantiles=ref_quantiles,
                ref_values=ref_values,
                blend_amount=channel_blend_amount[channel]
            )
        matched[..., channel] = matched_channel

    return matched.astype(np.float32, copy=False)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def setup_color_correction(numpy_image):
    print("Calibrating color correction.")
    return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2LAB)

def apply_color_correction(correction, numpy_image):
    print("Applying color correction.")
    lab_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2LAB)
    lab_correction = color_correct_lab_image(
        lab_image=lab_image,
        lab_correction=correction
    )
    image_rgb = cv2.cvtColor(lab_correction, cv2.COLOR_LAB2RGB)
    #  0.0-1.0 -> 0-255
    uint8_img = 255. * image_rgb
    return Image.fromarray(uint8_img.astype(np.uint8))

def load_img_numpy(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    # 0-255 -> 0.0-1.0
    image_numpy = np.asarray(image).astype(np.float32) / 255.0
    color_correction = setup_color_correction(image_numpy.copy())
    return (image_numpy, color_correction)

def load_img_torch(numpy_image):
    image = numpy_image[None].transpose(0, 3, 1, 2)
    torch_image = torch.from_numpy(image) 
    # 0.0-1.0 -> -1.0-1.0
    return 2.*torch_image - 1.

def save_image(torch_images, color_correction, sample_path, scale, strength, steps, seed):
    # -1.0-1.0 -> 0.0-1.0
    torch_images = torch.clamp((torch_images + 1.0) / 2.0, min=0.0, max=1.0)
    for torch_image in torch_images:
        numpy_image = torch_image.cpu().numpy()
        numpy_image = rearrange(numpy_image, 'c h w -> h w c').astype(np.float32)
        corrected_image = apply_color_correction(color_correction, numpy_image)
        img_name = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        img_path = os.path.join(sample_path, f"{img_name}_scale-{scale}_steps-{steps}_strenght-{strength}_seed-{seed}.png")
        corrected_image.save(img_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many subsequent batches",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="5.0",
        help="Comma separated list of scales to use"
    )
    parser.add_argument(
        "--strenghts",
        type=str,
        default="0.75",
        help="Comma separated list of strenghts to use",
    )
    parser.add_argument(
        "--waits",
        type=str,
        default="5,15,30",
        help="Wait times in seconds: after single image, after three images, after 9 images",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="stable-diffusion/configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="sd-v1-4.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="",
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--save_middle",
        action='store_true',
        help="Save also intermediate images",
        default=False
    )
    parser.add_argument(
        "--image_per_loop",
        action='store_true',
        help="Run one loop as single image",
        default=False
    )

    opt = parser.parse_args()
    
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    # Load model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    os.makedirs(opt.outdir, exist_ok=True)
    sample_path = opt.outdir

    prompt = opt.prompt
    assert prompt is not None
    prompts = [prompt]

    uc = model.get_learned_conditioning([""])                   
    c = model.get_learned_conditioning(prompts)

    assert os.path.isfile(opt.init_img)
    init_image_numpy, color_correction = load_img_numpy(opt.init_img)

    scales = [float(s) for s in opt.scales.split(',')]
    # unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
    strenghts_str = opt.strenghts or ','.join(['0.5'] * len(scales))
    strenghts = [float(s) for s in strenghts_str.split(',')]
    # strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image"
    scaling_options = list(zip(scales, strenghts))

    wait_single, wait_three, wait_nine = [int(s) for s in opt.waits.split(',')]
    
    max_loops = opt.n_samples
    static_seed = None
    if opt.seed:
        static_seed = int(opt.seed)
        max_loops = 1

    ddim_steps = opt.ddim_steps

    loops = 0
    images = 0
    while loops < max_loops:
        print(f'Loop {loops}')

        # Random seed and init sampler
        seed = static_seed or random.randint(0, 1_000_000_000)
        seed_everything(seed)
        sampler = DDIMSampler(model)
        sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

        # Reload image every loop to avoid mutation
        init_image_torch = load_img_torch(init_image_numpy).to(device)
        torch_images = repeat(init_image_torch, '1 ... -> b ...', b=1)

        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    for scale, strength in scaling_options:
                        # Don't overheat the GPU ;)
                        if images > 0:
                            sleep_seconds = wait_single
                            if images % 9 == 0:
                                sleep_seconds = wait_nine
                            elif images % 3 == 0:
                                sleep_seconds = wait_three
                            print(f'Sleeping for {sleep_seconds} seconds')
                            time.sleep(sleep_seconds)
                            print('Done')

                        print(f'Scale: {scale}, strenght: {strength}')

                        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
                        t_enc = int(strength * ddim_steps)
                        print(f"target t_enc is {t_enc} steps")

                        if opt.image_per_loop:
                            # Image get fed back again and again
                            pass
                        else:
                            torch_images = repeat(init_image_torch, '1 ... -> b ...', b=1)
                        
                        # First stage encoding (latent space)
                        current_latent = model.get_first_stage_encoding(model.encode_first_stage(torch_images))

                        # Second stage encoding (scaled latent)
                        z_enc = sampler.stochastic_encode(current_latent, torch.tensor([t_enc]).to(device))
                        
                        # Second stage decoding
                        samples = sampler.decode(
                            z_enc,
                            c,
                            t_enc,
                            unconditional_guidance_scale=scale,
                            unconditional_conditioning=uc
                        )

                        # First stage decoding
                        torch_images = model.decode_first_stage(samples)

                        # Clamp after every image to avoid color banding
                        torch_images = torch.clamp(torch_images, min=-1.0, max=1.0)

                        if opt.save_middle or not opt.image_per_loop:
                            save_image(
                                torch_images=torch_images,
                                color_correction=color_correction,
                                sample_path=sample_path,
                                scale=scale,
                                strength=strength,
                                steps=ddim_steps,
                                seed=seed
                            )
                        images = images + 1

                    # Save result after all scaling options have been added
                    if opt.image_per_loop and not opt.save_middle:
                        save_image(
                            torch_images=torch_images,
                            color_correction=color_correction,
                            sample_path=sample_path,
                            scale=scale,
                            strength=strength,
                            steps=ddim_steps,
                            seed=seed
                        )

        loops = loops + 1
    print("Done")

if __name__ == "__main__":
    main()
