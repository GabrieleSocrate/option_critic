"""This class stores all of the samples for training.  It is able to
construct randomly selected batches of phi's from the stored history.
"""

import numpy as np
import time
import theano

floatX = theano.config.floatX

class DataSet(object):
  """A replay memory consisting of circular buffers for observed images,
actions, and rewards.
  """
  """
  È una replay memory: una memoria circolare dove salvi le esperienze passate per allenare il critic con mini-batch casuali.
  Memorizza, per ogni step:
  l’immagine osservata (un frame)
  l’azione (qui nel tuo caso “azione” = opzione)
  il reward
  se l’episodio è finito (terminal)
  È “circolare” perché quando è piena, sovrascrive i dati più vecchi.
  """
  def __init__(self, width, height, rng, max_steps=1000, phi_length=4):
    """Construct a DataSet.

    Arguments:
      width, height - image size
      max_steps - the number of time steps to store
      phi_length - number of images to concatenate into a state
      rng - initialized numpy random number generator, used to
      choose random minibatches

    """
    # TODO: Specify capacity in number of state transitions, not
    # number of saved time steps.

    # Store arguments.
    self.width = width
    self.height = height
    self.max_steps = max_steps
    self.phi_length = phi_length
    self.rng = rng

    # Allocate the circular buffers and indices.
    self.imgs = np.zeros((max_steps, height, width), dtype='uint8')
    self.actions = np.zeros(max_steps, dtype='int32')
    self.rewards = np.zeros(max_steps, dtype=floatX)
    self.terminal = np.zeros(max_steps, dtype='bool')
    
    self.bottom = 0
    self.top = 0
    self.size = 0

  def add_sample(self, img, action, reward, terminal):
    """Add a time step record.

    Arguments:
      img -- observed image
      action -- action chosen by the agent
      reward -- reward received after taking the action
      terminal -- boolean indicating whether the episode ended
      after this time step
    """
    # salva tutto alla posizione top 
    self.imgs[self.top] = img
    self.actions[self.top] = action
    self.rewards[self.top] = reward
    self.terminal[self.top] = terminal

    # ora gestisce il caso di buffer pieno
    if self.size == self.max_steps:
      self.bottom = (self.bottom + 1) % self.max_steps
      # se è pieno allora sposti il bottom avanti di 1
    else:
      self.size += 1
      # altrimenti aumenti la size di 1
    self.top = (self.top + 1) % self.max_steps # aumenta il top di 1

  def __len__(self):
    """Return an approximate count of stored state transitions."""
    # TODO: Properly account for indices which can't be used, as in
    # random_batch's check.
    return max(0, self.size - self.phi_length)

  def last_phi(self):
    """Return the most recent phi (sequence of image frames). prendendoli direttamente dal buffer imgs"""
    indexes = np.arange(self.top - self.phi_length, self.top)
    return self.imgs.take(indexes, axis=0, mode='wrap')

  def phi(self, img):
    """Return a phi (sequence of image frames), using the last phi_length -
    1, plus img.
    """
    """
    Scopo generale: costruisce uno stato phi in cui l'ultimo frame è img e i primi phi-length-1 frame sono i più recenti già in memoria
    """
    indexes = np.arange(self.top - self.phi_length + 1, self.top)

    phi = np.empty((self.phi_length, self.height, self.width), dtype=floatX) 
    phi[0:self.phi_length - 1] = self.imgs.take(indexes,
                          axis=0,
                          mode='wrap')
    phi[-1] = img
    return phi

  def random_batch(self, batch_size, random_selection=False):
    """Return corresponding states, actions, rewards, terminal status, and
next_states for batch_size randomly chosen state transitions.

    """
    # Allocate the response.
    states = np.zeros((batch_size,
               self.phi_length,
               self.height,
               self.width),
              dtype='uint8')
    actions = np.zeros((batch_size), dtype='int32')
    rewards = np.zeros((batch_size), dtype=floatX)
    terminal = np.zeros((batch_size), dtype='bool')
    next_states = np.zeros((batch_size,
                self.phi_length,
                self.height,
                self.width),
                 dtype='uint8')

    count = 0
    indices = np.zeros((batch_size), dtype='int32')
      
    while count < batch_size: # loop finchè non riempi il batch
      # Randomly choose a time step from the replay memory.
      index = self.rng.randint(self.bottom,
                     self.bottom + self.size - self.phi_length) # sceglie un indice casuale nel replay

      initial_indices = np.arange(index, index + self.phi_length)
      transition_indices = initial_indices + 1
      end_index = index + self.phi_length - 1
      """
      Se phi_length=4:
      initial_indices = [t-3, t-2, t-1, t] → frame dello stato s_t
      transition_indices = [t-2, t-1, t, t+1] → frame dello stato successivo s_t+1
      end_index = t (l’ultimo frame dello stack)
      Quindi:
      lo stato è uno stack di 4 frame consecutivi
      il next_state è lo stack successivo shiftato di 1 frame
      """

      if np.any(self.terminal.take(initial_indices[0:-1], mode='wrap')):
        continue
      # controlla se nei primi 3 frame dello stack c'è un terminal = True, se c'è allora scarta il campione (questo perchè non vuoi che uno stato includa frame di due diversi episodi)

      indices[count] = index

      # Add the state transition to the response.
      states[count] = self.imgs.take(initial_indices, axis=0, mode='wrap')
      actions[count] = self.actions.take(end_index, mode='wrap')
      rewards[count] = self.rewards.take(end_index, mode='wrap')
      terminal[count] = self.terminal.take(end_index, mode='wrap')
      next_states[count] = self.imgs.take(transition_indices,
                        axis=0,
                        mode='wrap')
      count += 1

    return states, actions, rewards, next_states, terminal

if __name__ == "__main__":
  pass