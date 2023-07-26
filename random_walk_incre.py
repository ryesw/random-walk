import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
matplotlib.style.use('ggplot')
%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (10, 7)
from func import NUM_ALL_STATES, NUM_STATES, init_v_func, run_incre_mc_episode

num_episodes=int(1e5)
alpha=1e-4
V_TRUE = np.array([1/6, 2/6, 3/6, 4/6, 5/6])

print('working on {0}'.format(num_episodes))
v_func = init_v_func(NUM_ALL_STATES)
v_func = run_incre_mc_episode(v_func, num_episodes, alpha=alpha)

error=mean_squared_error(V_TRUE, v_func[1:-1])
print("Error: ",error)

# v_func[1:-1] no need to plot value of EXIT_STATES
plt.plot(range(1, NUM_STATES + 1), v_func[1:-1], 'o-', lw=1.5,
          label='num_episodes = {0}'.format(num_episodes))

# plot theorectical line
plt.title("Incremental Monte Carlo Prediction")
plt.plot(range(1, NUM_STATES + 1), V_TRUE, '--', color='black',
         label='theoretical')
plt.xlabel("States")
plt.ylabel("Value")

plt.legend(loc='best')
plt.xlim(0, NUM_STATES + 1)
plt.ylim(0, 1)