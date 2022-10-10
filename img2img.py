"""make variations of input image"""
from importlib import import_module
import datetime, time
import random
import argparse, os
import PIL
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


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


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
    init_image = load_img(opt.init_img).to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=1)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

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

                        # encode (scaled latent)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(device))
                        # decode it
                        samples = sampler.decode(
                            z_enc,
                            c,
                            t_enc,
                            unconditional_guidance_scale=scale,
                            unconditional_conditioning=uc
                        )

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            img_name = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
                            img.save(os.path.join(sample_path, f"{img_name}_scale-{scale}_steps-{ddim_steps}_strenght-{strength}_seed-{seed}.png"))
                        images = images + 1

        loops = loops + 1
    print("Done")

if __name__ == "__main__":
    main()
