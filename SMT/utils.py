import os
import logging
import time
from pathlib import Path

import progressbar

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def validate(training_iterations, logger, configs, habitat_configs, agent):
	horizon = configuration.TASK.HARIZON

	sum_reward = 0
	num_episodes = len(eval_nvironment.env.episodes)
	step = num_episodes//100

	bar = progressbar.ProgressBar(maxval=100, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()
	habitat_config.defrost()
	habitat_config.DATASET.DATA_PATH = '/data/datasets/pointnav/gibson/v1/val_mini/val_mini.json.gz'
	habitat_config.freeze()
	agent.environment.get_env().reconfigure(habitat_config)

	num_episodes = len(agent.environment.get_env().episodes)

	for e in range(0, num_episodes):
		# Reset the enviroment
		#print("EPISODE ", e)
		episode_reward = 0

		agent.reset() #reset the environment, sets the episode-index to e

		for timestep in range(horizon-1):
			action = agent.sample_action(evaluating=True)
			episode_reward += agent.step(action, timestep=timestep, training=False, evaluating=True)    

		sum_reward += episode_reward 
		
		if e%step == 0:
			bar.update(e//step + 1)

	bar.finish()
    
	logger.info('Validation reward for %i training iterations: %f' % (training_iterations, sum_reward/num_episodes))