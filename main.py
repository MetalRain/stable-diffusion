import argparse

from diffusion.worker import DiffusionTaskOptions
from diffusion.scheduler import schedule_task

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
        "--image",
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
        "--seed",
        type=str,
        default="",
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--save_middle",
        help="Save also intermediate images",
        default=""
    )
    parser.add_argument(
        "--plms",
        help="Use PLMS sampler",
        default=""
    )
    parser.add_argument(
        "--image_per_loop",
        help="Run one loop as single image",
        default=""
    )
    parser.add_argument(
        "--loop_to_loop",
        help="Continue same image from loop to next",
        default=""
    )
    parser.add_argument(
        "--task",
        help="Which task to run? Either 'txt2img' or 'img2img'",
    )

    opt = parser.parse_args()

    print('Running as control')
    print('Sending task to worker')
    args_dict=vars(opt)
    task_options = DiffusionTaskOptions(**args_dict)
    schedule_task(task_options)

if __name__ == "__main__":
    main()
