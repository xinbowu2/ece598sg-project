import sys
import glob
from navigator import *

def main_navigation(navigator, environment, mode, keyframes, keyframe_coordinates, keyframe_actions):
  print(CURRENT_NAVIGATION_ENVIRONMENT, environment)
  print(CURRENT_NAVIGATION_MODE, mode)
  game = test_setup()
  results = []
  for trial_index in range(NUMBER_OF_TRIALS):
    print('Trial index:', trial_index)
    movie_filename = '%s_%s_%d.mov' % (environment, mode, trial_index)
    
    goal_frame = game.get_state().goal_observation
    goal_location = game.get_state().goal_position

    goal_localization_keyframe_index = navigator.setup_navigation_test(game, keyframes, keyframe_coordinates, keyframe_actions, goal_frame, goal_location, movie_filename, environment)
    goal_localization_distance = get_distance(goal_location,
                                              keyframe_coordinates[goal_localization_keyframe_index])
    print('Localization distance:', goal_localization_distance)
    if mode == 'explore':
      navigator.show_memory_to_exploration_policy(keyframes)
    while not navigator.check_termination():
      print('completed:', 100 * float(navigator.get_steps()) / float(max_number_of_steps), '%')
      if mode == 'policy':
        navigator.policy_navigation_step()
      elif mode == 'random':
        navigator.random_explore_step()
      elif mode == 'explore':
        navigator.policy_explore_step()
      elif mode == 'teach_and_repeat':
        navigator.policy_navigation_step(teach_and_repeat=True)
      else:
        raise Exception('Please provide the mode: policy or random or explore!')
    results.append((navigator.check_goal_reached(),
                    navigator.get_steps(),
                    goal_localization_distance))
    print(results[-1])
    navigator.save_recordings()
  game.new_episode()
  print(FINAL_RESULTS, results)
  number_of_successes = sum([first for first, _, _ in results])
  print('Average success:', float(number_of_successes) / float(len(results)))
  print('Average success path length:', float(sum([first * second for first, second, _ in results])) / float(max(1, number_of_successes)))
  print('Average goal localization distance:', float(sum(third for _, _, third in results)) / float(len(results)))

if __name__ == '__main__':
  environment, mode = sys.argv[1], sys.argv[2]
  navigator = Navigator()

  trajectory_dir = '../trajectories/Adrian'

  images = []
  image_paths = []
  for im_path in glob.glob(trajectory_dir + "/images/*.png"):
    image_paths.append(im_path)
  image_paths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
  images = [center_crop_resize(plt.imread(x)[:,:,:3], 256) for x in image_paths][:-1]
  actions = np.load(trajectory_dir + '/actions.npy', allow_pickle=True)[:-1]
  positions = np.load(trajectory_dir + '/positions.npy', allow_pickle=True)[:-1]
  positions = np.array([positions[:,2], positions[:,0]]).T
  assert len(images) == len(actions)+1 == len(positions), 'Length of inputs not the same'

  print('Starting navigation!')
  main_navigation(navigator, environment, mode, images, positions, actions)
  print('Navigation finished!')
