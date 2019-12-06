from train.train_setup import *
import habitat
import pdb
import random

def data_generator():
	config = habitat.get_config(config_file='datasets/pointnav/gibson.yaml')
	config.defrost()
	config.DATASET.SPLIT = 'train_mini'
	config.ENVIRONMENT.MAX_EPISODE_STEPS = MAX_CONTINUOUS_PLAY*10
	#config.SEED = random.randint(1, ACTION_MAX_EPOCHS)
	config.freeze()
	# print(config)
	env = habitat.Env(config=config)
	random.shuffle(env.episodes)
	action_mapping = {      
		0: 'move_forward',
		1: 'turn left',
		2: 'turn right',
		3: 'stop'
	}

	while True:
		x_result = []
		y_result = []
		for episode in range(EDGE_EPISODES):
			current_x = center_crop_resize(env.reset()['rgb']/255.0, 64)

			x = []
			for _ in range(MAX_CONTINUOUS_PLAY):
				action_index = random.randint(0, len(action_mapping)-2)
				current_y = action_index
				x.append(current_x)
				current_x = center_crop_resize(env.step(action_index)['rgb']/255.0, 64)
					
				if env.episode_over:
					current_x = center_crop_resize(env.reset()['rgb']/255.0, 64)
					break
			first_second_label = []
			current_first = 0
			while True:
				y = None
				current_second = None
				if random.random() < 0.5:
					y = 1
					second = current_first + random.randint(1, MAX_ACTION_DISTANCE)
					if second >= MAX_CONTINUOUS_PLAY:
						break
					current_second = second
				else:
					y = 0
					second = current_first + random.randint(1, MAX_ACTION_DISTANCE)
					if second >= MAX_CONTINUOUS_PLAY:
						break
					current_second_before = None
					current_second_after = None
					index_before_max = current_first - NEGATIVE_SAMPLE_MULTIPLIER * MAX_ACTION_DISTANCE
					index_after_min = current_first + NEGATIVE_SAMPLE_MULTIPLIER * MAX_ACTION_DISTANCE
					if index_before_max >= 0:
						current_second_before = random.randint(0, index_before_max)
					if index_after_min < MAX_CONTINUOUS_PLAY:
						current_second_after = random.randint(index_after_min, MAX_CONTINUOUS_PLAY - 1)
					if current_second_before is None:
						current_second = current_second_after
					elif current_second_after is None:
						current_second = current_second_before
					else:
						if random.random() < 0.5:
							current_second = current_second_before
						else:
							current_second = current_second_after
				first_second_label.append((current_first, current_second, y))
				current_first = second + 1
			random.shuffle(first_second_label)
			for first, second, y in first_second_label:
				future_x = x[second]
				current_x = x[first]
				current_y = y
				x_result.append(np.concatenate((current_x, future_x), axis=2))
				y_result.append(current_y)
		number_of_batches = int(len(x_result) / BATCH_SIZE)
		for batch_index in range(number_of_batches):
			from_index = batch_index * BATCH_SIZE
			to_index = (batch_index + 1) * BATCH_SIZE
			x_out = np.array(x_result[from_index:to_index])
			yield (x_out, np.array(y_result[from_index:to_index]))

if __name__ == '__main__':
	logs_path, current_model_path = setup_training_paths('../experiments/edge/default_experiment')
	model = SiameseResnet(EDGE_CLASSES)
	model.build((32, 64, 64, 6))
	model.load_weights("../experiments/edge/experiment1/models/model.000250.h5")
	#model = EDGE_NETWORK(((1 + EDGE_STATE_ENCODING_FRAMES) * NET_CHANNELS, NET_HEIGHT, NET_WIDTH), EDGE_CLASSES)
	adam = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	callbacks_list = [tf.keras.callbacks.TensorBoard(log_dir=logs_path, write_graph=False),
										tf.keras.callbacks.ModelCheckpoint(current_model_path,
										period=MODEL_CHECKPOINT_PERIOD)]
	model.fit_generator(data_generator(),
						steps_per_epoch=DUMP_AFTER_BATCHES,
						epochs=EDGE_MAX_EPOCHS,
						callbacks=callbacks_list)
