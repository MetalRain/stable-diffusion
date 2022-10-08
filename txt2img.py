"""Make images"""
from importlib import import_module
import datetime, time
import argparse, os
import random
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

# from ldm.util import instantiate_from_config
ldm_util = import_module("stable-diffusion.ldm.util")
instantiate_from_config = ldm_util.instantiate_from_config

# from ldm.models.diffusion.ddim import DDIMSampler
plms = import_module("stable-diffusion.ldm.models.diffusion.plms")
PLMSSampler = plms.PLMSSampler


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
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
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
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many subsequent batches",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="7.5",
        help="Comma separated list of scales to use"
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="",
        help="Use static seed, forces n_samples to 1"
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

    scales = [float(s) for s in opt.scales.split(',')]
    # unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))

    max_loops = opt.n_samples
    static_seed = None
    if opt.seed:
        static_seed = opt.seed
        max_loops = 1

    loops = 0
    while loops < max_loops:
        print(f'Loop {loops}')

        if loops > 0:
            if loops % 9 == 0:
                print('Sleeping for 5 seconds')
                time.sleep(5)
                print('Done')
            elif loops % 3 == 0:
                print('Sleeping for 2 seconds')
                time.sleep(2)
                print('Done')

        # Random seed and init sampler
        seed = static_seed or random.randint(0, 10000000)
        seed_everything(seed)
        sampler = PLMSSampler(model)
        start_code = torch.randn([1, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

        precision_scope = autocast if opt.precision=="autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():

                    for scale in scales:
                        ddim_steps = 10 + int(scale * 5)

                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = sampler.sample(
                            S=ddim_steps,
                            conditioning=c,
                            batch_size=1,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=scale,
                            unconditional_conditioning=uc,
                            eta=0.0,
                            x_T=start_code
                        )

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_checked_image_torch = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_checked_image_torch:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            img_name = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
                            img.save(os.path.join(sample_path, f"{img_name}_scale-{scale}_steps-{ddim_steps}_seed-{seed}.png"))
        loops = loops + 1
    print("Done")

if __name__ == "__main__":
    main()
