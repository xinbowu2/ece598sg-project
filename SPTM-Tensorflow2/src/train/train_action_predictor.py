from train.train_setup import *
import pdb
import habitat
import random
print("IMPORTS COMPLETE")

def data_generator():
  config = habitat.get_config(config_file='datasets/pointnav/gibson.yaml')
  config.defrost()
  config.DATASET.SPLIT = 'train_mini'
  config.ENVIRONMENT.MAX_EPISODE_STEPS = MAX_CONTINUOUS_PLAY*64
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

  current_x = center_crop_resize(env.reset()['rgb']/255.0, 256)
  # print(config)
  yield_count = 0
  while True:
    if yield_count >= ACTION_MAX_YIELD_COUNT_BEFORE_RESTART:
      current_x = center_crop_resize(env.reset()['rgb']/255.0, 256)
      yield_count = 0
    x = []
    y = []
    
    for _ in range(MAX_CONTINUOUS_PLAY):
      #current_x = game.get_state().screen_buffer.transpose(VIZDOOM_TO_TF)
      action_index = random.randint(0, len(action_mapping)-2)
      # game_make_action_wrapper(game, ACTIONS_LIST[action_index], TRAIN_REPEAT)
      current_y = action_index
      x.append(current_x)
      y.append(current_y)
      current_x = center_crop_resize(env.step(action_index)['rgb']/255.0, 256)
        
      if env.episode_over:
        current_x = center_crop_resize(env.reset()['rgb']/255.0, 256)
        break

    first_second_pairs = []
    current_first = 0
    while True:
      distance = random.randint(1, MAX_ACTION_DISTANCE)
      second = current_first + distance
      if second >= min(len(x), MAX_CONTINUOUS_PLAY):
        break
      first_second_pairs.append((current_first, second))
      current_first = second + 1
    random.shuffle(first_second_pairs)
    x_result = []
    y_result = []
    for first, second in first_second_pairs:
      future_x = x[second]
      current_x = x[first]
      previous_x = current_x
      if first > 0:
        previous_x = x[first - 1]
      current_y = y[first]
      x_result.append(np.concatenate((previous_x, current_x, future_x), axis=2))
      y_result.append(current_y)
      if len(x_result) == BATCH_SIZE:
        yield_count += 1 
        print(np.array(x_result).shape)
        yield (np.array(x_result),
               np.array(y_result))
        x_result = []
        y_result = []
  env.close()

if __name__ == '__main__':
  print("HELLOOOO")
  logs_path, current_model_path = setup_training_paths(EXPERIMENT_OUTPUT_FOLDER)
  print(logs_path, current_model_path)

  #model = ACTION_NETWORK(((1 + ACTION_STATE_ENCODING_FRAMES) * NET_CHANNELS, NET_HEIGHT, NET_WIDTH), ACTION_CLASSES)
  model = ResNet18(3)
  #model.build((16, 256, 256, 9))
  #model.load_weights("../experiments/experiment1/models/model.000200.h5")
  adam = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
  model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
  callbacks_list = [tf.keras.callbacks.TensorBoard(log_dir=logs_path, write_graph=False),
                    tf.keras.callbacks.ModelCheckpoint(current_model_path,
                                                    period=MODEL_CHECKPOINT_PERIOD)]
  model.fit_generator(data_generator(),
                      steps_per_epoch=DUMP_AFTER_BATCHES,
                      epochs=ACTION_MAX_EPOCHS,
                      callbacks=callbacks_list)
