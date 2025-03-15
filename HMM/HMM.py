
# Code modified from BE3M33UI - Artificial Intelligence course

import random
from collections import Counter
from itertools import product
import itertools
import matplotlib.pyplot as plt
import bisect
from PIL import Image, ImageDraw
from matplotlib.animation import FuncAnimation

random.seed(0)

NORTH = (-1, 0)
EAST = (0, 1)
SOUTH = (1, 0)
WEST = (0, -1)
ALL_DIRS = (NORTH, EAST, SOUTH, WEST)

DEFAULT_MOVE_PROBS = {
    NORTH: 0.25,
    EAST: 0.25,
    SOUTH: 0.25,
    WEST: 0.25
}

def weighted_random_choice(pdist):
    """Choose an element from distribution given as dict of values and their probs."""
    choices, weights = zip(*pdist.items())
    return random.choices(choices, weights=weights, k=1)[0]

def normalized(P, factor=None, return_normalization_factor=False):
    """Return a normalized copy of the distribution given as a Counter"""
    if not factor:
        s = sum(P.values())
        factor = 1 / s
    norm = Counter({k: factor*v for k, v in P.items()})
    if return_normalization_factor:
        return norm, factor
    else:
        return norm

def add(s, dir):
    """Add direction to a state"""
    return tuple(a+b for a, b in zip(s, dir))


def direction(s1, s2):
    """Return a direction vector from s1 to s2"""
    return tuple(b-a for a, b in zip(s1, s2))


def manhattan(s1, s2):
    return sum(abs(a) for a in direction(s1, s2))

## ------------------------------------------
class Maze:
    """Representation of a maze"""

    WALL_CHAR = '#'
    FREE_CHAR = ' '

    def __init__(self, map):
        self.load_map(map)
        self._free_positions = []
        self._grid = []
        

    def load_map(self, map_fname):
        """Load map from a text file"""
        self.map = []
        with open(map_fname, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                self.map.append(line)
        self.map = tuple(self.map)
        self.height = len(self.map)
        self.width = len(self.map[0])
        self.min_pos = (1, 1)
        self.max_pos = (len(self.map)-2, len(self.map[0])-2)

    def __str__(self):
        """Return a string representation of the maze"""
        return '\n'.join(self.map)

    def is_free(self, pos):
        """Check whether a position is free
        
        :param pos: position as (row, column)
        :return: True or False
        """
        return self.map[pos[0]][pos[1]] == self.FREE_CHAR

    def is_wall(self, pos):
        """Check whether a position contains a wall

        :param pos: position as (row, column)
        :return: True or False
        """
        return self.map[pos[0]][pos[1]] == self.WALL_CHAR

    def get_free_positions(self, search=False):
        """Return a list of all free positions in the maze
        
        It returns a cached list, if the free positions were already precomputed.
        """
        # If free positions already found and shouldn't find them anew
        if self._free_positions and not search:
            return self._free_positions
        fp = []
        for r in range(1, self.height-1):
            for c in range(1, self.width-1):
                pos = (r, c)
                if self.is_free(pos):
                    fp.append(pos)
        self._free_positions = fp
        return fp

    def get_wall_positions(self):
        """Return a list of all maze positions containing walls"""
        wp = []
        for r in range(0, self.height):
            for c in range(0, self.width):
                pos = (r, c)
                if self.is_wall(pos):
                    wp.append(pos)
        return wp

    def get_dist_to_wall(self, pos, dir):
        """Return the distance to the nearest wall.
        
        Distance is the number of steps one can make from
        the given position to the closest wall in the given direction.
        
        :param pos: (row, column) tuple 
        :param dir: One of the following:
                    (-1, 0) ... north
                    (0, 1) ... east
                    (1, 0) ... south
                    (0, -1) ... west
        :return: int, distance to wall
        """
        if pos not in self.get_free_positions():
            raise ValueError('The specified robot position is not allowable. Did you use (row, column)?')
        d = 0
        while True:
            pos = add(pos, dir)
            if not self.is_free(pos):
                return d
            d += 1

    def get_grid(self, search=False):
        """Displays the maze
        """
        # If free positions already found and shouldn't find them anew
        #if self._grid and not search:
        #    return self._grid
        grid = []
        for r in range(self.height):
            grow = []
            for c in range(self.width):
                pos = (r, c)
                if self.is_free(pos):
                    grow.append(1)
                else:
                    grow.append(0)
            grid.append(grow)
        self._grid = grid
        return grid

    def display1(self, prob=None, state=None):
    
      grid = self.get_grid()  
      plt.imshow(self.get_grid(), cmap='gray', interpolation='nearest')

      if prob:
        max_prob = max(prob.values())  # Find the maximum probability
        for pos in self.get_free_positions():
          grid[pos[0]][pos[1]] = grid[pos[0]][pos[1]] * prob[pos]/ max_prob
        plt.imshow(grid, cmap='Purples', alpha=.7, interpolation='nearest')

      if state:
        plt.text(state[1],state[0], '*')

    # uses pillow
    def display2(self, prob=None, state=None):
        # Create a new image with white background
        grid = self.get_grid()
        height, width = len(grid), len(grid[0])
        img = Image.new('RGBA', (width * 10, height * 10), 'white')
        draw = ImageDraw.Draw(img)

        # Draw the maze
        for r in range(height):
            for c in range(width):
                color = 'black' if grid[r][c] == 0 else 'white'
                draw.rectangle([c * 10, r * 10, (c + 1) * 10, (r + 1) * 10], fill=color)

        # Draw probabilities if provided
        if prob:
            max_prob = max(prob.values())
            for pos in self.get_free_positions():
                intensity = int(255 * prob[pos] / max_prob)
                draw.rectangle([pos[1] * 10, pos[0] * 10, (pos[1] + 1) * 10, (pos[0] + 1) * 10], fill=(0, intensity, 0, 150))

        # Draw the robot
        if state:
            robot_color = (255, 0, 0, 150)
            draw.ellipse([state[1] * 10 + 3, state[0] * 10 + 3, state[1] * 10 + 7, state[0] * 10 + 7], fill=robot_color)

        # Show the image
        plt.imshow(img)
        plt.axis('off')  # Hide axes
  
    # for displaying the particles
    def display3(self, particles, state=None):
      """Display the maze with particles and the robot's state."""
      # Create a new image with white background
      grid = self.get_grid()
      height, width = len(grid), len(grid[0])
      img = Image.new('RGBA', (width * 10, height * 10), 'white')
      draw = ImageDraw.Draw(img)

      # Draw the maze
      for r in range(height):
          for c in range(width):
              color = 'black' if grid[r][c] == 0 else 'white'
              draw.rectangle([c * 10, r * 10, (c + 1) * 10, (r + 1) * 10], fill=color)

      # Draw particles
      particle_color = (0, 0, 0, 255)  
      particle_count = Counter(particles)
      for particle, count in particle_count.items():
          for _ in range(count):
              x_offset = random.randint(0, 9)
              y_offset = random.randint(0, 9)
              draw.point([particle[1] * 10 + x_offset, particle[0] * 10 + y_offset], fill=particle_color)

      # Draw the robot
      if state:
          robot_color = (255, 0, 255, 150)
          draw.ellipse([state[1] * 10 + 3, state[0] * 10 + 3, state[1] * 10 + 7, state[0] * 10 + 7], fill=robot_color)

      # Show the image
      plt.imshow(img)
      plt.axis('off')  # Hide axes
      

class NearFarSensor:
    """Crude sensor measuring direction"""

    VALUES = ['n', 'f']
    DIST_OF_SURE_FAR = 4  # For this distance and larger, the sensor will surely return 'f'

    def __init__(self, robot, direction):
        """Initialize sensor
        
        :param robot: Robot, to which this sensor belongs
        :param direction: direction, 2-tuple, of the sensor
        """
        self.robot = robot
        self.dir = direction

    def get_value_probabilities(self):
        """Return the probabilities of individual observations depending on robot position"""
        dist = self.robot.get_dist_to_wall(self.dir)
        p = {}
        p['n'] = max([1 - dist/self.DIST_OF_SURE_FAR, 0])
        p['f'] = 1 - p['n']
        return p

    def read(self):
        """Return a single sensor reading depending on robot position"""
        p = self.get_value_probabilities()
        return weighted_random_choice(p)
    

## -------------------------------------
class Robot():
  """Robot in a maze as HMM"""

  def __init__(self, sensor_directions=None, move_probs=None):
      """Initialize robot with sensors and transition model
      
      :param sensor_directions: list of directions of individual sensors
      :param move_probs: distribution over move directions 
      """
      self.maze = None
      self.position = None
      if not sensor_directions:
          sensor_directions = ALL_DIRS
      self.sensors = []
      for dir in sensor_directions:
          self.sensors.append(NearFarSensor(robot=self, direction=dir))
      self.move_probs = move_probs if move_probs else DEFAULT_MOVE_PROBS

  def observe(self, state=None):
      """Perform single observation of all sensors
      
      :param state: robot state (position) for which the observation
                    shall be made. If no state is given, the current 
                    robot position is used.
      :return: tuple of individual sensor readings
      """
      if not state:
          return tuple(s.read() for s in self.sensors)
      saved_pos = self.position
      self.position = state
      obs = self.observe()
      self.position = saved_pos
      return obs

  def _next_move_dir(self):
      """Return the direction of next move"""
      return weighted_random_choice(self.move_probs)

  def get_dist_to_wall(self, dir, pos=None):
      """Return the distance to wall"""
      if not pos:
          pos = self.position
      return self.maze.get_dist_to_wall(pos, dir)

  def get_states(self):
      """Return the list of possible states"""
      return self.maze.get_free_positions()

  def get_targets(self, state):
      """Return the list of all states reachable in one step from the given state"""
      tgts = [state]
      for dir in ALL_DIRS:
          next_state = add(state, dir)
          if not self.maze.is_free(next_state): continue
          tgts.append(next_state)
      return tgts

  def get_observations(self):
      """Return the list of all possible observations"""
      sensor_domains = [s.VALUES for s in self.sensors]
      return list(product(*sensor_domains))

  def get_next_state_distr(self, cur_state):
      """Return the distribution over possible next states
      
      Takes the walls around current state into account.
      """
      p = Counter()
      for dir in ALL_DIRS:
          next_state = add(cur_state, dir)
          if not self.maze.is_free(next_state):
              pass
          else:
              p[next_state] = self.move_probs[dir]
      return normalized(p)

  def pt(self, cur_state, next_state):
      """Return a single transition probability"""
      p = self.get_next_state_distr(cur_state)
      return p[next_state]

  def pe(self, pos, obs):
      """Return the probability of observing obs in state pos"""
      # Store current robot position and set a new one
      stored_pos = self.position
      self.position = pos
      # Compute the probability of observation
      p = 1
      for sensor, value in zip(self.sensors, obs):
          pd = sensor.get_value_probabilities()
          p *= pd[value]
      # Restore robot position
      self.position = stored_pos
      return p

  def set_random_position(self):
      """Set the robot to a random admissible state"""
      self.position = random.choice(self.maze.get_free_positions())

  def step(self, state=None):
      """Generate a next state for the current state"""
      if not state:
          state = self.position
      next_dist = {next_state: self.pt(state, next_state)
                    for next_state in self.get_targets(state)}
      # Sample from the distribution
      next_pos =  weighted_random_choice(next_dist)
      self.position = next_pos
      return next_pos
  
  def sample_step(self, state=None):
      """Generate a next state for the current state"""
      if not state:
          state = self.position
      next_dist = {next_state: self.pt(state, next_state)
                    for next_state in self.get_targets(state)}
      # Sample from the distribution
      next_pos =  weighted_random_choice(next_dist)
      return next_pos
  
  def simulate(self, init_state=None, n_steps=5):
      """Perform several simulation steps starting from the given initial state

      :return: 2-tuple, sequence of states, and sequence of observations
      """
      if not init_state:
          init_state = self.position
      return self._simulate(init_state, n_steps)

  def _simulate(self, init_state, n_steps):
      """Perform several simulation steps starting from the given initial state

      :return: 2-tuple, sequence of states, and sequence of observations
      """
      last_state = init_state
      states, observations = [], []
      for i in range(n_steps):
          state = self.step(last_state)
          observation = self.observe(state)
          states.append(state)
          observations.append(observation)
          last_state = state
      return states, observations


# input robot: robot in the maze instance
#       filt_prob: previous estimate of the location (P(X_t|e_1:t))
#       observation: current observation vector       
def forward(robot, filt_prob, observation):
  all_free_pos = robot.maze.get_free_positions()
  
  ## Prediction
  pred = {}
  for x1 in all_free_pos:
    pred[x1] = 0
    for x0 in all_free_pos:
      pred[x1] += robot.pt(x0, x1) * filt_prob[x0]

  ## Filtering
  for x1 in all_free_pos:
    filt_prob[x1] = robot.pe(x1, observation) * pred[x1]

  # Normalize
  s = sum([value for value in filt_prob.values()])

  # normalize the vector
  filt_prob = {key: value / s for key, value in filt_prob.items()}

  return filt_prob

def particle_filter(robot, particles, observation):
    """Perform particle filtering to estimate the robot's position."""
    
    # Weights for each particle
    weights = [1.0 / num_particles] * num_particles
    
    # Predict new particle positions using transition probabilities
    predicted_particles = [robot.sample_step(particle) for particle in particles]
    
    # Update weights based on the observation
    for i, particle in enumerate(predicted_particles):
        weights[i] *= robot.pe(particle, observation)
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        # If all weights are zero, reinitialize particles
        particles = [random.choice(robot.maze.get_free_positions()) for _ in range(num_particles)]
        weights = [1.0 / num_particles] * num_particles
    else:
        weights = [w / total_weight for w in weights]
    
    # Resample particles based on weights
    new_particles = random.choices(predicted_particles, weights, k=num_particles)
    
    return new_particles

# -----------------------------------------------------
robot = Robot()
#robot.maze = Maze('./mazes/rect_3x2_empty.map')
robot.maze = Maze('mazes/rect_4x12_maze.map')
robot.position = (1, 1) # start position


# ---------------------- Exact Inference -------------
# uniform distribution for all free positions (prior belief)
total_free_positions = len(robot.maze.get_free_positions())
filt_prob = {x:1/total_free_positions for x in robot.maze.get_free_positions()}
state = robot.position
  
# Initialize the plot
fig, ax = plt.subplots()

def update1(frame):
    """Update function for animation."""
    global state, filt_prob

    if (frame):
      # Simulate one step
      state = robot.step()
      observation = robot.observe(state)

      # Update the filtering probabilities
      filt_prob = forward(robot, filt_prob, observation)

    # Display the maze with updated probabilities
    ax.clear()
    robot.maze.display2(filt_prob, state)

# ----------------------- Particle filtering based Inference -------------
# Example usage of particle_filter
num_particles = 20
# Initialize particles randomly across all free positions
particles = [random.choice(robot.maze.get_free_positions()) for _ in range(num_particles)]
    
state = robot.position
def update2(frame):
    """Update function for animation using particle filter."""
    global state, particles
    
    if frame:
        # Simulate one step
        state = robot.step()
        observation = robot.observe(state)

        # Estimate the position using particle filter
        particles = particle_filter(robot, particles, observation)
        #print(f"Estimated Position: {estimated_position}, Actual Position: {state}")

    # Display the maze with estimated position
    ax.clear()
    robot.maze.display3(particles, state)

    
# use update1 for exact inference, and update2 for particle filter based    
ani = FuncAnimation(fig, update2, frames=100, interval=100, repeat=False)
ani.save('animation.mp4', writer='ffmpeg', fps=5)
plt.show()