import numpy as np


STARTING_STATE = 3
NUM_STATES = 5
STATES = np.arange(1, NUM_STATES + 1)
# STATE id happens to be the same as the true denormalized state value.

# Add 2 exit state, the state after exit action, whose state value will be 0
EXIT_STATES = [0, NUM_STATES + 1]
NUM_ALL_STATES = NUM_STATES + len(EXIT_STATES)


def is_exit_state(state):
    return state in EXIT_STATES

def is_reward_exit_state(state):
    return state == EXIT_STATES[-1]
    # 입력 s0 출력 transition

def take_action(s0):
    rand = np.random.random()
    if rand >= 0.5:
        s1 = s0 - 1
    else:
        s1 = s0 + 1
        
    if is_reward_exit_state(s1):
        reward = 1
    else:
        reward = 0
    return s1, reward

def init_v_func(num_all_states):
    """initialize value function"""
    v_func = np.zeros(num_all_states)
    # The value of EXIT_STATES should be 0
    return v_func

# Temporal Difference
def run_td_episode(v_func, s0=STARTING_STATE, alpha=None, gamma=1.0):
  next_state, reward = s0, 0

  states = [s0]
  rewards = []
  next_states = []
  next_state = s0

  while True:
    next_state, reward = take_action(next_state)
    rewards.append(reward)
    next_states.append(next_state)
    states.append(next_state)

    if is_exit_state(next_state):
      states.pop()
      break
  
  iter = zip(states, rewards, next_states)
  for state, reward, next_state in iter:
    td_target = reward + gamma * v_func[int(next_state)]
    v_func[int(state)] += alpha * (td_target - v_func[int(state)])

  return v_func

# Monte Carlo
def run_mc_episode(v_func, num_episodes, s0=STARTING_STATE, gamma=1.0):
  n_v = np.zeros(NUM_ALL_STATES) # 상태 s를 방문한 횟수
  s_v = np.zeros(NUM_ALL_STATES) # 상태 s에 대한 return들의 합

  # 하나의 episode에 대해 episode의 총 개수만큼 계산
  for _ in range(num_episodes):

    states = [s0]
    rewards = []
    next_state = s0

    while True:
      next_state, reward = take_action(next_state)
      rewards.append(reward)
      if not(is_exit_state(next_state)):
        states.append(next_state)
      else:
        break
    
    states = reversed(states)
    rewards = reversed(rewards)
    iter = zip(states, rewards)
    cum_r = 0
    for s, r in iter:
      cum_r *= gamma
      cum_r += r

      n_v[int(s)] += 1
      s_v[int(s)] += cum_r

  v_func = s_v / (n_v )
  return v_func