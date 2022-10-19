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
from pytorch_lightning import seed_everything

# from ldm.util import instantiate_from_config
ldm_util = import_module("stable-diffusion.ldm.util")
instantiate_from_config = ldm_util.instantiate_from_config

# from ldm.models.diffusion.ddim import DDIMSampler
ddim = import_module("stable-diffusion.ldm.models.diffusion.ddim")
DDIMSampler = ddim.DDIMSampler


# Color correction from skimage
# https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/exposure/histogram_matching.py#L24-L85

def cdf_prep(template):
    """Prepare template values for cdf calculation"""
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)
    # calculate normalized quantiles for each array
    tmpl_quantiles = np.cumsum(tmpl_counts) / float(template.size)
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
    if inv_blend < 1.0:
        return (
            source.ravel() * blend_amount
            + new_values * inv_blend
        ).reshape(source.shape)
    else:
        return new_values.reshape(source.shape)


def color_correct_lab_image(lab_image, lab_correction, transform_count, strength):
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
    # Keep L channel as is, no color correction
    # For A and B use color correction
    channel_blend_iterations = [1, 2, 2]
    channel_blend_amount=[1.0, strength, strength]
    # In order to combat against darkening, boost L +2 every time color corrected
    channel_value_boost = [1.0 * transform_count, 0.0, 0.0]
    channel_value_multiplier = [1.0, 1.0, 1.0]
    
    matched = np.empty(lab_image.shape, dtype=lab_image.dtype)
    
    # Adjust channels
    for channel in range(lab_image.shape[-1]):
        source_channel = lab_image[..., channel]
        
        matched_channel = source_channel
        # multiply channel values
        matched_channel = np.clip(
            (matched_channel * channel_value_multiplier[channel]) + channel_value_boost[channel],
            # Clamp CIELAB 0-100 for L and -150-150 for A and B
            0.0 if channel == 0 else -150.0,
            100.0 if channel == 0 else 150.0,
        )
        # Blend several times to soften otherwise hard changes
        if channel_blend_amount[channel] < 1.0:
            reference_channel = lab_correction[..., channel]
            # Use same ref for all rounds
            ref_quantiles, ref_values = cdf_prep(
                template=reference_channel
            )
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

def apply_color_correction(correction, numpy_image, transform_count, strength):
    print("Applying color correction.")
    lab_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2LAB)
    lab_correction = color_correct_lab_image(
        lab_image=lab_image,
        lab_correction=correction,
        transform_count=transform_count,
        strength=strength
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
    return 2.0 * torch_image - 1.0

class DiffusionRunner:
    def __init__(self, config_path, checkpoint_path):
        config = OmegaConf.load(f"{config_path}")
        model = load_model_from_config(config, f"{checkpoint_path}")

        # Load model
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model.to(self.device)

    def run_task(self, task):
        uc = self.model.get_learned_conditioning([""])
        c = self.model.get_learned_conditioning([task.prompt])

        wait_single, wait_three, wait_nine = task.waits

        static_seed = task.static_seed
        max_loops = task.max_loops
        if static_seed:
            max_loops = 1

        loops = 0
        images = 0
        while loops < max_loops:
            print(f'Loop {loops}')

            # Random seed and init sampler
            seed = static_seed or random.randint(0, 1_000_000_000)
            seed_everything(seed)
            sampler = DDIMSampler(self.model)

            init_values = task.loop_init(self.device, sampler)
            torch_images = None

            precision_scope = autocast
            with torch.no_grad():
                with precision_scope("cuda"):
                    with self.model.ema_scope():
                        for scaling in task.scaling_options:
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

                            print(f'Scaling: {scaling}')

                            torch_images = task.transform(
                                model=self.model,
                                device=self.device,
                                sampler=sampler,
                                init_values=init_values,
                                torch_images=torch_images,
                                conditioning=c,
                                unconditional_conditioning=uc,
                                scaling=scaling,
                            )

                            # Clamp after every image to avoid color banding
                            torch_images = torch.clamp(torch_images, min=-1.0, max=1.0)

                            task.loop_middle(
                                torch_images=torch_images,
                                scaling=scaling,
                                seed=seed
                            )
                            images = images + 1

                        task.loop_end(
                            torch_images=torch_images,
                            scaling=scaling,
                            seed=seed
                        )
            loops = loops + 1

class DiffusionTask:
    def loop_init(
            self,
            device,
            sampler):
        return ()

    def transform(
            self,
            model,
            device,
            sampler,
            init_values,
            torch_images,
            conditioning,
            unconditional_conditioning,
            scaling):
        return None

    def loop_middle(
            self,
            torch_images,
            scaling,
            seed):
        pass

    def loop_end(
            self,
            torch_images,
            scaling,
            seed):
        pass


class DiffusionImg2ImgTask(DiffusionTask):
    def __init__(self, opt):
        os.makedirs(opt.outdir, exist_ok=True)
        self.sample_path = opt.outdir

        self.prompt = opt.prompt

        assert os.path.isfile(opt.init_img)
        init_image_numpy, color_correction = load_img_numpy(opt.init_img)
        self.original_image = init_image_numpy
        self.color_reference = color_correction

        scales = [float(s) for s in opt.scales.split(',')]
        # unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
        strenghts_str = opt.strenghts or ','.join(['0.5'] * len(scales))
        strenghts = [float(s) for s in strenghts_str.split(',')]
        # strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image"
        self.scaling_options = list(zip(scales, strenghts))

        self.waits = [int(s) for s in opt.waits.split(',')]

        self.transforms_without_cc = 0
        
        self.max_loops = opt.n_samples
        self.static_seed = int(opt.seed) if opt.seed else None
        self.ddim_steps = opt.ddim_steps
        self.image_per_loop = opt.image_per_loop
        self.save_middle = opt.save_middle

    def loop_init(self, device, sampler):
        sampler.make_schedule(ddim_num_steps=self.ddim_steps, ddim_eta=0.0, verbose=False)
        # Reload image every loop to avoid mutation
        self.original_image_torch = load_img_torch(self.original_image).to(device)
        torch_images = repeat(self.original_image_torch, '1 ... -> b ...', b=1)
        self.transforms_without_cc = 0
        
        return torch_images

    def transform(
            self,
            model,
            device,
            sampler,
            init_values,
            torch_images,
            conditioning,
            unconditional_conditioning,
            scaling):
        scale, strength = scaling

        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(strength * self.ddim_steps)
        print(f"target t_enc is {t_enc} steps")

        if self.image_per_loop:
            # Image get fed back again and again
            torch_images = torch_images if torch_images is not None else init_values
        else:
            torch_images = repeat(self.original_image_torch, '1 ... -> b ...', b=1)
            self.transforms_without_cc = 0
        
        # First stage encoding (latent space)
        current_latent = model.get_first_stage_encoding(model.encode_first_stage(torch_images))

        # Second stage encoding (scaled latent)
        z_enc = sampler.stochastic_encode(current_latent, torch.tensor([t_enc]).to(device))
        
        # Second stage decoding
        samples = sampler.decode(
            z_enc,
            conditioning,
            t_enc,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=unconditional_conditioning
        )

        # First stage decoding
        torch_images = model.decode_first_stage(samples)

        return torch_images

    def loop_middle(
            self,
            torch_images,
            scaling,
            seed):
        # Save after each image or if asked to save in middle
        if self.save_middle or not self.image_per_loop:
            self.save_image(
                torch_images,
                scaling,
                seed
            )
            self.transforms_without_cc = 0

    def loop_end(
            self,
            torch_images,
            scaling,
            seed):
        # Save result after all scaling options have been added
        if self.image_per_loop and not self.save_middle:
            self.save_image(
                torch_images,
                scaling,
                seed
            )
            self.transforms_without_cc = 0

    def save_image(
            self,
            torch_images, 
            scaling,
            seed):
        scale, strength = scaling
        # -1.0-1.0 -> 0.0-1.0
        torch_images = torch.clamp((torch_images + 1.0) / 2.0, min=0.0, max=1.0)
        for torch_image in torch_images:
            numpy_image = torch_image.cpu().numpy()
            numpy_image = rearrange(numpy_image, 'c h w -> h w c').astype(np.float32)
            corrected_image = apply_color_correction(self.color_reference, numpy_image, self.transforms_without_cc, strength)
            img_name = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
            img_path = os.path.join(self.sample_path, f"{img_name}_scale-{scale}_steps-{self.ddim_steps}_strenght-{strength}_seed-{seed}.png")
            corrected_image.save(img_path)

class DiffusionText2ImgTask:
    def __init__(self, opt):
        self.prompt = opt.prompt

        os.makedirs(opt.outdir, exist_ok=True)
        self.sample_path = opt.outdir

        scales = [float(s) for s in opt.scales.split(',')]
        # unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))

        self.scaling_options = list(scales)

        self.waits = [int(s) for s in opt.waits.split(',')]
        self.C = 4
        self.f = 8
        self.W = opt.W
        self.H = opt.H
        
        self.max_loops = opt.n_samples
        self.static_seed = int(opt.seed) if opt.seed else None

    def loop_init(
            self,
            device,
            sampler):
        start_code = torch.randn([1, self.C, self.H // self.f, self.W // self.f], device=device)
        return start_code
    
    def transform(
            self,
            model,
            device,
            sampler,
            init_values,
            torch_images,
            conditioning,
            unconditional_conditioning,
            scaling):
        scale = scaling
        self.ddim_steps = 10 + int(scale * 5)

        shape = [self.C, self.H // self.f, self.W // self.f]
        start_code = init_values
        samples_ddim, _ = sampler.sample(
            S=self.ddim_steps,
            conditioning=conditioning,
            batch_size=1,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=unconditional_conditioning,
            eta=0.0,
            x_T=start_code
        )
        
        torch_images = model.decode_first_stage(samples_ddim)
        return torch_images

    def loop_middle(
            self,
            torch_images,
            scaling,
            seed):
        self.save_image(
            torch_images, 
            scaling,
            seed
        )

    def loop_end(
            self,
            torch_images,
            scaling,
            seed):
        pass

    def save_image(
            self,
            torch_images, 
            scaling,
            seed):
        scale = scaling
        # -1.0-1.0 -> 0.0-1.0
        torch_images = torch.clamp((torch_images + 1.0) / 2.0, min=0.0, max=1.0)
        for torch_image in torch_images:
            numpy_image = 255. * rearrange(torch_image.cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(numpy_image.astype(np.uint8))
            img_name = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
            img.save(os.path.join(self.sample_path, f"{img_name}_scale-{scale}_steps-{self.ddim_steps}_seed-{seed}.png"))


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
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
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
        default="",
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
    parser.add_argument(
        "--task",
        help="Which task to run? Either 'txt2img' or 'img2img'",
    )

    opt = parser.parse_args()
    
    task = None
    if opt.task == 'img2img':
        task = DiffusionImg2ImgTask(opt)
    elif opt.task == 'txt2img':
        task = DiffusionText2ImgTask(opt)

    if task:
        runner = DiffusionRunner(opt.config, opt.ckpt)
        runner.run_task(task)
    else:
        print('No task, select with --task')

if __name__ == "__main__":
    main()
