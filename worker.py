import argparse
import time

from diffusion.worker import DiffusionRunner, DiffusionImg2ImgTask, DiffusionText2ImgTask
from diffusion.db import fetch_task, complete_task

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--waits",
        type=str,
        default="1,5,10",
        help="Wait times in seconds: after single image, after three images, after 9 images",
    )

    opt = parser.parse_args()

    print('Running as worker')
    runner = None
    while True:
        print('Fetching task')
        task_options = fetch_task()
        if task_options:
            if not runner:
                print('Starting runner')
                runner = DiffusionRunner(task_options.config, task_options.ckpt, opt.waits)
            print(f'Starting task {task_options.task_id}..')
            task = None
            if task_options.task == 'img2img':
                task = DiffusionImg2ImgTask(task_options)
            elif task_options.task == 'txt2img':
                task = DiffusionText2ImgTask(task_options)
            
            if task:
                runner.run_task(task)
                complete_task(task_options)
                print(f'Task {task_options.task_id} done!')
        time.sleep(1)

if __name__ == "__main__":
    main()
