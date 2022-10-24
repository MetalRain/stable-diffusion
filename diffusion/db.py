import sqlite3
import uuid
from datetime import datetime

db_uri = 'file:task_db?mode=rwc'

def connect():
    db = sqlite3.connect(db_uri, uri=True)
    db.row_factory = sqlite3.Row
    return db

def ensure_tables():
    db = connect()
    try:
        db.execute('''
        CREATE TABLE images (
            id VARCHAR(36),
            parent_image_id VARCHAR(36),
            task_id VARCHAR(36),
            width INTEGER,
            height INTEGER,
            filepath VARCHAR(1000),
            seed INTEGER,
            scale REAL,
            strength REAL
        );
        ''')
        db.execute('''
        CREATE TABLE prompts (
            id VARCHAR(36),
            prompt VARCHAR(2000)
        );
        ''')
        db.execute('''
        CREATE TABLE task_groups (
            id VARCHAR(36)
        );
        ''')
        db.execute('''
        CREATE TABLE tasks (
            id VARCHAR(36),
            task_group_id VARCHAR(36),
            input_image_id VARCHAR(36),
            width INTEGER,
            height INTEGER,
            output_folder VARCHAR(1000),
            prompt_id VARCHAR(36),
            seed INTEGER,
            sampler VARCHAR(100),
            sampler_steps INTEGER,
            iterations INTEGER,
            merge_loops INTEGER,
            merge_steps INTEGER,
            save_intermediates INTEGER,
            program TEXT,
            completed_at VARCHAR(19)
        );
        ''')
        db.commit()
    except sqlite3.OperationalError as exc:
        print(exc)
    return db

def create_task_group():
    db = connect()
    try:
        # create
        row = dict(
            id=str(uuid.uuid4()),
        )
        print('Task group', row)
        db.execute(
            'INSERT INTO task_groups (id) VALUES (?);',
            list(row.values())
        )
        db.commit()
        return row
    except sqlite3.OperationalError as exc:
        print(exc)
    return None

def ensure_prompt(prompt_text):
    db = connect()
    try:
        # Find by prompt
        query = '''
        SELECT id, prompt
        FROM prompts
        WHERE prompt = ?
        '''
        res = db.execute(query, (prompt_text,))
        row = res.fetchone()
        if row:
            return row
        # Or create
        row = dict(
            id=str(uuid.uuid4()),
            prompt=prompt_text
        )
        print('Prompt', row)
        db.execute(
            'INSERT INTO prompts (id, prompt) VALUES (?, ?);',
            list(row.values())
        )
        db.commit()
        return row
    except sqlite3.OperationalError as exc:
        print(exc)
    return None

def ensure_image(filepath, width, height, task_id):
    db = connect()
    try:
        # Find by filepath
        query = '''
        SELECT
            id, parent_image_id, task_id, width, height,
            filepath, seed, scale, strength
        FROM images
        WHERE filepath = ?
        '''
        res = db.execute(query, (filepath, ))
        row = res.fetchone()
        if row:
            return row
        # Or create new
        row = dict(
            id=str(uuid.uuid4()),
            parent_image_id=None,
            task_id=task_id,
            width=width,
            height=height,
            filepath=filepath,
            seed=None,
            scale=None,
            strength=None
        )
        print('Image', row)
        query = '''
        INSERT INTO images
            (id, parent_image_id, task_id, width, height,
            filepath, seed, scale, strength)
        VALUES (?, ?, ?, ?, ?,
            ?, ?, ?, ?);
        '''
        db.execute(
            query,
            list(row.values())
        )
        db.commit()
        return row
    except sqlite3.OperationalError as exc:
        print(exc)
    return None

def save_task(task_options, task_group_id=None):
    db = ensure_tables()
    task_id=str(uuid.uuid4())
    db_prompt = ensure_prompt(task_options.prompt)
    db_image = None
    if task_options.image:
        db_image = ensure_image(task_options.image, task_options.W, task_options.H, task_id)

    if task_group_id is None:
        db_task_group = create_task_group()
        task_group_id = db_task_group['id']

    scale_values = task_options.scales.split(',')
    strenght_values = task_options.strenghts.split(',') or len(scale_values) * [1.0]
    row = dict(
        id=task_id,
        task_group_id=task_group_id,
        input_image_id=db_image['id'] if db_image else None,
        width=db_image['width'] if db_image else task_options.W,
        height=db_image['height'] if db_image else task_options.H,
        output_folder=task_options.outdir,
        prompt_id=db_prompt['id'],
        seed=task_options.seed,
        sampler='plms' if task_options.plms else 'ddim',
        sampler_steps=task_options.ddim_steps,
        iterations=task_options.n_samples,
        merge_loops=1 if task_options.loop_to_loop else 0,
        merge_steps=1 if task_options.image_per_loop else 0,
        save_intermediates=1 if task_options.save_middle else 0,
        program='|'.join([
            f'{ scale };{ strenght }'
            for scale, strenght in zip(scale_values, strenght_values)
        ]),
        completed_at=None
    )
    print('Task', row)

    try:
        query = '''
        INSERT INTO tasks
            (id, task_group_id, input_image_id, width, height, output_folder, prompt_id,
            seed, sampler, sampler_steps, iterations,
            merge_loops, merge_steps, save_intermediates,
            program, completed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?);
        '''
        db.execute(
            query,
            list(row.values())
        )
        db.commit()
        return row
    except sqlite3.OperationalError as exc:
        print(exc)
    return None

def fetch_task():
    from diffusion.worker import DiffusionTaskOptions

    db = ensure_tables()
    try:
        query = '''
        SELECT id, task_group_id, input_image_id, width, height, output_folder, prompt_id,
            seed, sampler, sampler_steps, iterations,
            merge_loops, merge_steps, save_intermediates,
            program, completed_at
        FROM tasks
        WHERE id IN (
            SELECT id
            FROM tasks
            WHERE completed_at IS NULL
            ORDER BY RANDOM()
            LIMIT 10
        )
        '''
        res = db.execute(query)
        task_row = res.fetchone()
        if not task_row:
            print('Tasks not found')
            return None

        query = '''
        SELECT id, prompt
        FROM prompts
        WHERE id = ?
        '''
        res = db.execute(query, (task_row['prompt_id'], ))
        prompt_row = res.fetchone()
        if not prompt_row:
            print('Prompt not found')
            return None
        
        image_row = None
        if task_row['input_image_id']:
            query = '''
            SELECT id, parent_image_id, width, height, filepath
            FROM images
            WHERE id = ?
            '''
            res = db.execute(query, (task_row['input_image_id'], ))
            image_row = res.fetchone()

        scales = []
        strenghts = []
        program_steps = task_row['program'].split('|')
        program_pairs = [tuple(step.split(';')) for step in program_steps]
        for scale, strenght in program_pairs:
            scales.append(scale)
            strenghts.append(strenght)
        
        task_options = DiffusionTaskOptions(
            prompt=prompt_row['prompt'],
            outdir=task_row['output_folder'],
            scales=",".join(scales),
            task='img2img' if image_row else 'txt2img',
            seed=task_row['seed'],
            strenghts=",".join(strenghts),
            ddim_steps=task_row['sampler_steps'],
            n_samples=task_row['iterations'],
            image=image_row['filepath'] if image_row else None,
            H=image_row['height'] if image_row else task_row['height'],
            W=image_row['width'] if image_row else task_row['width'],
            save_middle=task_row['save_intermediates'] == 1,
            plms=task_row['sampler'] == 'plms',
            image_per_loop=task_row['merge_steps'] == 1,
            loop_to_loop=task_row['merge_loops'] == 1,
            task_id=task_row['id'],
            image_id=image_row['id'] if image_row else None
        )
        return task_options
    except sqlite3.OperationalError as exc:
        print(exc)
    return None

def complete_task(task_options):
    db = connect()
    try:
        query = '''
        UPDATE tasks
        SET completed_at = ?
        WHERE id = ?
        '''
        db.execute(
            query,
            (datetime.now().isoformat()[:19], task_options.task_id,)
        )
        db.commit()
        return True
    except sqlite3.OperationalError as exc:
        print(exc)
    return False
    