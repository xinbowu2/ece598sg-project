import progressbar


def validate(training_iterations, configs, val_environment, agent):
	horizon = configuration.TASK.HARIZON

	sum_reward = 0
	num_episodes = len(eval_nvironment.env.episodes)

	for e in range(0, num_episodes):
		# Reset the enviroment
		print("EPISODE ", e)
		episode_reward = 0

		agent.reset(e) #reset the environment, sets the episode-index to e

		bar = progressbar.ProgressBar(maxval=horizon/10, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
		bar.start()

		for timestep in range(horizon):
			action = agent.sample_action(validating=True)
			episode_reward += agent.step(action, training=False)    

			if timestep%10 == 0:
				bar.update(timestep/10 + 1)
		sum_reward += episode_reward 
		
		bar.finish()

	 print('Validation reward for %i training iterations: %i' % (training_iterations, sum_reward/num_episodes))