from test_setup import *
import os.path

from numpy import mean
from numpy import median
import networkx as nx

from trajectory_plotter import *

def top_number_to_threshold(n, top_number, values):
  top_number = min([top_number, n])
  threshold = np.percentile(values, (n - top_number) * 100 / float(n))
  return threshold

def sieve(shortcuts, top_number):
  if top_number == 0:
    return []
  probabilities = shortcuts[:, 0]
  n = shortcuts.shape[0]
  threshold = top_number_to_threshold(n, top_number, probabilities)
  print('Confidence threshold for top', top_number, 'out of', n, ':', threshold)
  sieved_shortcut_indexes = []
  for index in xrange(n):
    if probabilities[index] >= threshold:
      sieved_shortcut_indexes.append(index)
  return shortcuts[sieved_shortcut_indexes]

class InputProcessor:
  def __init__(self):
    self.edge_model = SiameseResnet(EDGE_CLASSES)
    self.edge_model.build((32, 256, 256, 6))
    self.edge_model.load_weights('../edge_model.h5')
    self.edge_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    self.siamese = True
    self.bottom_network = build_bottom_network(self.edge_model, (NET_CHANNELS, NET_HEIGHT, NET_WIDTH))
    self.top_network = build_top_network(self.edge_model)

  def preprocess_input(self, input):
    return input

  def set_memory_buffer(self, keyframes):
    keyframes = [self.preprocess_input(keyframe) for keyframe in keyframes]
  
    memory_codes = self.bottom_network.predict(np.array(keyframes))
    list_to_predict = []
    for index in xrange(len(keyframes)):
      x = np.concatenate((memory_codes[0], memory_codes[index]), axis=0)
      list_to_predict.append(x)
    self.tensor_to_predict = np.array(list_to_predict)

  def append_to_memory_buffer(self, keyframe):
    keyframe = self.preprocess_input(keyframe)

    memory_code = self.bottom_network.predict(expanded_keyframe)
    x = np.concatenate((memory_code, memory_code), axis=1)
    self.tensor_to_predict = np.concatenate((self.tensor_to_predict, x), axis=0)

  def predict_single_input(self, input):
    input = self.preprocess_input(input)
  
    input_code = np.squeeze(self.bottom_network.predict(np.expand_dims(input, axis=0), batch_size=1))
    for index in xrange(self.tensor_to_predict.shape[0]):
      self.tensor_to_predict[index][0:(input_code.shape[0])] = input_code
    probabilities = self.top_network.predict(self.tensor_to_predict,
                                              batch_size=TESTING_BATCH_SIZE)
    return probabilities[:, 1]

  def get_memory_size(self):
    return self.tensor_to_predict.shape[0]

class SPTM:
  def __init__(self):
    self.input_processor = InputProcessor()

  def set_shortcuts_cache_file(self, environment):
    # if no limit, MEMORY_MAX_FRAMES is None
    if MEMORY_MAX_FRAMES is None:
      max_frames = -1
    else:
      max_frames = MEMORY_MAX_FRAMES
    self.shortcuts_cache_file = SHORTCUTS_CACHE_FILE_TEMPLATE % (environment, MEMORY_SUBSAMPLING, max_frames)

  def set_memory_buffer(self, keyframes):
    self.input_processor.set_memory_buffer(keyframes)

  def append_to_memory_buffer(self, keyframe):
    self.input_processor.append_to_memory_buffer(keyframe)

  def predict_single_input(self, input):
    return self.input_processor.predict_single_input(input)

  def get_memory_size(self):
    return self.input_processor.get_memory_size()

  def add_double_sided_edge(self, first, second):
    self.graph.add_edge(first, second)
    self.graph.add_edge(second, first)

  def add_double_forward_biased_edge(self, first, second):
    self.graph.add_edge(first, second)
    self.graph.add_edge(second, first, {'weight' : 1000000000})

  def smooth_shortcuts_matrix(self, shortcuts_matrix, keyframe_coordinates):
    for first in xrange(len(shortcuts_matrix)):
      for second in xrange(first + 1, len(shortcuts_matrix)):
        shortcuts_matrix[first][second] = (shortcuts_matrix[first][second] + 
                                           shortcuts_matrix[second][first]) / 2.0
    shortcuts = []
    for first in xrange(len(shortcuts_matrix)):
      for second in xrange(first + 1 + MIN_SHORTCUT_DISTANCE, len(shortcuts_matrix)):
        values = []
        for shift in xrange(-SHORTCUT_WINDOW, SHORTCUT_WINDOW + 1):
          first_shifted = first + shift
          second_shifted = second + shift
          if first_shifted < len(shortcuts_matrix) and second_shifted < len(shortcuts_matrix) and first_shifted >= 0 and second_shifted >= 0:
            values.append(shortcuts_matrix[first_shifted][second_shifted])
        quality = median(values)
        distance = get_distance(keyframe_coordinates[first],
                                keyframe_coordinates[second])
        shortcuts.append((quality, first, second, distance))
    return np.array(shortcuts)

  def compute_shortcuts(self, keyframes, keyframe_coordinates):
    self.set_memory_buffer(keyframes)
    if not os.path.isfile(self.shortcuts_cache_file):
      shortcuts_matrix = []
      for first in xrange(len(keyframes)):
        probabilities = self.predict_single_input(keyframes[first])
        shortcuts_matrix.append(probabilities)
        print('Finished:', float(first * 100) / float(len(keyframes)), '%')
      shortcuts = self.smooth_shortcuts_matrix(shortcuts_matrix, keyframe_coordinates)
      shortcuts = sieve(shortcuts, LARGE_SHORTCUTS_NUMBER)
      np.save(self.shortcuts_cache_file, shortcuts)
    else:
      shortcuts = np.load(self.shortcuts_cache_file)
    self.shortcuts = sieve(shortcuts, SMALL_SHORTCUTS_NUMBER)

  def get_number_of_shortcuts(self):
    return len(self.shortcuts)

  def get_shortcut(self, index):
    return (int(self.shortcuts[index, 1]), int(self.shortcuts[index, 2]))

  def get_shortcuts(self):
    return self.shortcuts

  def build_graph(self, keyframes, keyframe_coordinates):
    self.set_memory_buffer(keyframes)
    memory_size = self.get_memory_size()
    self.graph = nx.Graph()
    self.graph.add_nodes_from(range(memory_size))
    for first in xrange(memory_size - 1):
      # self.add_double_forward_biased_edge(first, first + 1)
      self.add_double_sided_edge(first, first + 1)
    self.compute_shortcuts(keyframes, keyframe_coordinates)
    for index in xrange(self.get_number_of_shortcuts()):
      edge = self.get_shortcut(index)
      first, second = edge
      assert abs(first - second) > MIN_SHORTCUT_DISTANCE
      self.add_double_sided_edge(*edge)

  def find_nn(self, input):
    probabilities = self.predict_single_input(input)
    best_index = np.argmax(probabilities)
    best_probability = np.max(probabilities)
    return best_index, best_probability, probabilities

  def set_goal(self, goal_frame, real_goal_coordinates, keyframe_coordinates):
    self.step = 0
    best_index, probabilities, nns = self.find_knn_median_threshold(goal_frame, NUMBER_OF_NEAREST_NEIGHBOURS, 0.0)
    print(nns)
    print([probabilities[nn] for nn in nns])
    print([get_distance(real_goal_coordinates, keyframe_coordinates[nn]) for nn in nns])
    print([keyframe_coordinates[nn] for nn in nns])
    print(real_goal_coordinates)
    best_probability = 1.0
    if best_index is None:
      best_index, best_probability, _ = self.find_nn(goal_frame)
    memory_size = self.get_memory_size()
    goal_index = memory_size
    self.graph.add_node(goal_index)
    edge = (best_index, goal_index)
    self.add_double_sided_edge(*edge)
    self.append_to_memory_buffer(goal_frame)
    print('Real goal distance:', get_distance(real_goal_coordinates, keyframe_coordinates[best_index]))
    self.smoothed_memory = None
    self.last_nn = None
    return best_index, best_probability

  def compute_shortest_paths(self, graph_goal):
    self.shortest_paths = nx.shortest_path(self.graph, target=graph_goal, weight='weight')
    self.shortest_distances = [len(value) - 1 for value in self.shortest_paths.values()]
    print('Mean shortest_distances to goal:', mean(self.shortest_distances))
    print('Median shortest_distances to goal:', median(self.shortest_distances))

  def get_shortest_paths_and_distances(self):
    return self.shortest_paths, self.shortest_distances

  def _find_neighbours_by_threshold(self, threshold, probabilities):
    nns = []
    for index, probability in enumerate(probabilities):
      if probability >= threshold:
        nns.append(index)
    return nns

  def find_neighbours_by_threshold(self, input, threshold):
    probabilities = self.predict_single_input(input)
    return self._find_neighbours_by_threshold(threshold, probabilities)

  def find_knn(self, input, k):
    probabilities = self.predict_single_input(input)
    threshold = top_number_to_threshold(self.get_memory_size(),
                                        k,
                                        probabilities)
    return self._find_neighbours_by_threshold(threshold, probabilities)

  def find_knn_median_threshold(self, input, k, threshold):
    probabilities = self.predict_single_input(input)
    knn_threshold = top_number_to_threshold(self.get_memory_size(),
                                            k,
                                            probabilities)
    final_threshold = max([threshold, knn_threshold])
    nns = self._find_neighbours_by_threshold(final_threshold, probabilities)
    nns.sort()
    if nns:
      nn = nns[len(nns) / 2]
      return nn, probabilities, nns
    else:
      return None, probabilities, nns

  def find_nn_threshold(self, input, threshold):
    nn, probability, probabilities = self.find_nn(input)
    if probability < threshold:
      return None, None
    else:
      return nn, probabilities

  def find_nn_on_last_shortest_path(self, input):
    if self.last_nn is None:
      return None, None
    probabilities = self.predict_single_input(input)
    last_shortest_path_prefix = self.shortest_paths[self.last_nn][:(MAX_LOOK_AHEAD + 1)]
    path_probabilities = np.array([probabilities[index] for index in last_shortest_path_prefix])
    best_look_ahead = np.argmax(path_probabilities)
    best_probability = np.max(path_probabilities)
    if best_probability < WEAK_INTERMEDIATE_REACHABLE_GOAL_THRESHOLD:
      return None, None
    return last_shortest_path_prefix[best_look_ahead], probabilities

  def find_smoothed_nn(self, input):
    nn = None
    if SMOOTHED_LOCALIZATION:
      nn, probabilities = self.find_nn_on_last_shortest_path(input)
    if nn is None:
      nn, probabilities, _ = self.find_knn_median_threshold(input, NUMBER_OF_NEAREST_NEIGHBOURS, INTERMEDIATE_REACHABLE_GOAL_THRESHOLD)
    return nn, probabilities

  def select_IRG_on_shortest_path(self, nn, probabilities):
    shortest_path = self.shortest_paths[nn]
    print('Current shortest path:', len(shortest_path) - 1)
    if self.plot_shortest_path:
      plotter = TrajectoryPlotter(os.path.join(EVALUATION_PATH, 'shortest_path%d_%d.pdf' % (self.trial_index, self.step)), *TEST_SETUPS[self.environment].box)
      self.step += 1
      for point in shortest_path:
        plotter.add_point(keyframe_coordinates[point][:2])
        plotter.add_edge((current_coordinates[0],
                          current_coordinates[0],
                          current_coordinates[1],
                          current_coordinates[1]))
      plotter.save()
      self.plot_shortest_path = False
    if SMOOTHED_LOCALIZATION:
      upper_limit = len(shortest_path) - 1
      valid_min_look_ahead = min(MIN_LOOK_AHEAD, upper_limit)
      valid_max_look_ahead = min(MAX_LOOK_AHEAD, upper_limit)
      best_look_ahead = valid_min_look_ahead
      for look_ahead in xrange(valid_min_look_ahead,
                               valid_max_look_ahead + 1):
        index = shortest_path[look_ahead]
        if probabilities[index] >= INTERMEDIATE_REACHABLE_GOAL_THRESHOLD:
          best_look_ahead = look_ahead
    else:
      best_look_ahead = 0
      for look_ahead, index in enumerate(shortest_path):
        if probabilities[index] >= INTERMEDIATE_REACHABLE_GOAL_THRESHOLD:
          best_look_ahead = look_ahead
    IRG = shortest_path[best_look_ahead]
    print('Found IRG:', IRG)
    return IRG

  def find_intermediate_reachable_goal(self, input, current_coordinates, keyframe_coordinates):
    nn, probabilities = self.find_smoothed_nn(input)
    self.last_nn = nn
    if nn is None:
      print('Found no IRG!')
      return None, None
    else:
      return self.select_IRG_on_shortest_path(nn, probabilities), nn
