from diffusion.db import save_task

def schedule_task(task_options, target_size=3):
    '''Split long tasks into size of target_size'''
    if task_options.loop_to_loop:
        save_task(task_options)
        return 1
    iterations = task_options.n_samples
    steps_in_program = len(task_options.scales)
    steps_in_task = iterations * steps_in_program
    tasks_scheduled = 0
    if steps_in_task > target_size:
        iterations_per_task = max(1, min(1, int(target_size / steps_in_program)))
        steps_per_task = iterations_per_task * steps_in_program
        tasks = int(steps_in_task / steps_per_task)
        task_options.n_samples = iterations_per_task
        task_group_id = None
        for i in range(tasks):
            task = save_task(task_options, task_group_id)
            tasks_scheduled += 1
            if task_group_id is None:
                task_group_id = task['task_group_id']
        # one short, add one
        if tasks * steps_per_task < steps_in_task:
            task_options.n_samples = 1 
            save_task(task_options, task_group_id)
            tasks_scheduled += 1
        return tasks_scheduled
    save_task(task_options)
    return 1