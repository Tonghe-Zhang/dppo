import numpy as np
def get_schedule(schedule: str):
    schedule_type = schedule.split('(')[0]
    schedule_args = schedule.split('(')[1].split(')')[0].split(',')
    if schedule_type == 'constant':
        assert len(schedule_args) == 1
        return lambda x: float(schedule_args[0])
    elif schedule_type == 'linear':
        assert len(schedule_args) == 4
        eps_max, eps_min, init_steps, anneal_steps = float(schedule_args[0]), float(schedule_args[1]), int(schedule_args[2]), int(schedule_args[3])
        return lambda x: np.clip(eps_max - (eps_max - eps_min) * (x - init_steps) / anneal_steps, eps_min, eps_max)
    elif schedule_type == 'cosine':
        assert len(schedule_args) == 4
        eps_max, eps_min, init_steps, anneal_steps = float(schedule_args[0]), float(schedule_args[1]), int(schedule_args[2]), int(schedule_args[3])
        return lambda x: eps_min + (eps_max - eps_min) * 0.5 * (1 + np.cos(np.clip((x - init_steps) / anneal_steps, 0, 1) * np.pi))
    else:
        raise ValueError('Unknown schedule: %s' % schedule_type)
def current_time():
    from datetime import datetime
    # Get current time
    now = datetime.now()
    # Format the time to the desired pattern
    formatted_time = now.strftime("%y-%m-%d-%H-%M-%S")
    return formatted_time
