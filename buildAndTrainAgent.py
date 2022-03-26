
# Commented out IPython magic to ensure Python compatibility.
# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Is this notebook running on Colab or Kaggle?
IS_COLAB = "google.colab" in sys.modules
IS_KAGGLE = "kaggle_secrets" in sys.modules

if IS_COLAB or IS_KAGGLE:
    !apt update && apt install -y libpq-dev libsdl2-dev swig xorg-dev xvfb
    !pip install -q -U tf-agents pyvirtualdisplay gym[atari,box2d]

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

if not tf.config.list_physical_devices('GPU'):
    print("No GPU was detected. CNNs can be very slow without a GPU.")
    if IS_COLAB:
        print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")
    if IS_KAGGLE:
        print("Go to Settings > Accelerator and select GPU.")

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# To get smooth animations
import matplotlib.animation as animation
mpl.rc('animation', html='jshtml')

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rl"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

from google.colab import drive
drive.mount('/content/drive')

!cp -r "/content/drive/MyDrive/Roms" /content/
!python -m atari_py.import_roms Roms

# import gym
# atariGameTitles=[]
# for z in gym.envs.registry.all():
#     envTitle=str(z).partition(')')[0].partition('(')[2]
#     if envTitle.count('-ram')>0:
#         if atariGameTitles.count(envTitle.partition('-ram')[0])==0:
#             atariGameTitles.append(envTitle.partition('-ram')[0])
# print('The Atari Games Available Are')
# print(atariGameTitles)
# for gameTitle in atariGameTitles:
#     if gameTitle != 'Defender':
#         #Defender either does not work or takes too long
#         myEnv=gym.make(gameTitle+'-v4')
#         print('Action Space for '+gameTitle+'-v4 is '+str(myEnv.action_space)+
#         ' corresponding to:')
#         print(myEnv.get_action_meanings())
#         print()

tf.random.set_seed(42)
np.random.seed(42)

import tf_agents
from tf_agents.environments import suite_gym

tf_agents.__version__

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim

env = suite_gym.load('Freeway-v4')
env

env.gym

import pyvirtualdisplay
!apt-get install -y xvfb x11-utils
display = pyvirtualdisplay.Display(visible=False, size=(1400, 900)).start()

img = env.render(mode="rgb_array")

plt.figure(figsize=(6, 8))
plt.imshow(img)
plt.axis("off")
save_fig("freeway_plot")
plt.show()

env.current_time_step()

env.observation_spec()

env.action_spec()

env.time_step_spec()

env.seed(42)
obs = env.reset()
for _ in range(5):
    time_step = env.step(0) # LEFT
    img = env.render(mode="rgb_array")

    plt.figure(figsize=(6, 8))
    plt.imshow(img)
    plt.axis("off")
    save_fig("breakout_plot")
    plt.show()

obs

from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
max_episode_steps = 27000 # <=> 108k ALE frames since 1 step = 4 frames
environment_name = "Freeway-v4"
env = suite_atari.load(
    environment_name,
    max_episode_steps=max_episode_steps,
    gym_env_wrappers=[AtariPreprocessing, FrameStack4])
from tf_agents.environments.tf_py_environment import TFPyEnvironment
tf_env = TFPyEnvironment(env)

eval_py_env = suite_gym.load('Freeway-v4')
eval_env = TFPyEnvironment(eval_py_env)

from tf_agents.networks.q_network import QNetwork
preprocessing_layer = keras.layers.Lambda(
lambda obs: tf.cast(obs, np.float32) / 255.)
conv_layer_params=[(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
fc_layer_params=[512]
q_net = QNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        preprocessing_layers=preprocessing_layer,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params
        )

from tf_agents.agents.dqn.dqn_agent import DqnAgent
# see TF-agents issue #113
#optimizer = keras.optimizers.RMSprop(lr=2.5e-4, rho=0.95, momentum=0.0,
# epsilon=0.00001, centered=True)
train_step = tf.Variable(0)
update_period = 4 # run a training step every 4 collect steps
optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=2.5e-4, decay=0.95,
momentum=0.0,epsilon=0.00001, centered=True)
epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
initial_learning_rate=1.0, # initial ?
decay_steps=250000 // update_period, # <=> 1,000,000 ALE frames
end_learning_rate=0.01) # final ?
agent = DqnAgent(tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        target_update_period=2000, # <=> 32,000 ALE frames
        td_errors_loss_fn=keras.losses.Huber(reduction="none"),
        gamma=0.99, # discount factor
        train_step_counter=train_step,
        epsilon_greedy=lambda: epsilon_fn(train_step))
agent.initialize()

from tf_agents.replay_buffers import tf_uniform_replay_buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=100000)
replay_buffer_observer = replay_buffer.add_batch

tempdir = "/content/drive/MyDrive/a4cp"
from tf_agents.policies import policy_saver
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from multiprocessing import Process

def f(train_checkpointer, global_step, tf_policy_saver, policy_dir):
    try:
        print('\n ---- checkpoint process started ----\n')
        train_checkpointer.save(global_step)
    except Exception as e:
        print(e)
    print("\n ---- checkpointed, saving policy ----\n")
    try:
        tf_policy_saver.save(policy_dir)
        print("\n ---- checkpoint process done ----\n")
    except Exception as e:
        print(e)
    return;

# checkpoint_dir = os.path.join(tempdir, 'checkpoint')
# train_checkpointer = common.Checkpointer(
#     ckpt_dir=checkpoint_dir,
#     max_to_keep=1,
#     agent=agent,
#     policy=agent.policy,
#     replay_buffer=replay_buffer,
#     global_step=train_step
# )
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=agent)
manager = tf.train.CheckpointManager(ckpt, tempdir + '/tf_ckpts', max_to_keep=3)

policy_dir = os.path.join(tempdir, 'policy')
tf_policy_saver = policy_saver.PolicySaver(agent.policy)
tf_policy_saver.save(policy_dir)

# train_checkpointer.initialize_or_restore()
# train_step = tf.compat.v1.train.get_global_step()

# saved_policy = tf.compat.v2.saved_model.load(policy_dir)
# run_episodes_and_create_video(saved_policy, eval_env, eval_py_env)

class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")

from tf_agents.metrics import tf_metrics
train_metrics = [
tf_metrics.NumberOfEpisodes(),
tf_metrics.EnvironmentSteps(),
tf_metrics.AverageReturnMetric(),
tf_metrics.AverageEpisodeLengthMetric(),
]

from tf_agents.eval.metric_utils import log_metrics
import logging
logging.getLogger().setLevel(logging.INFO)
log_metrics(train_metrics)

from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
collect_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=update_period)

from tf_agents.policies.random_tf_policy import RandomTFPolicy
initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),
tf_env.action_spec())
init_driver = DynamicStepDriver(
    tf_env,
    initial_collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=20000) # <=> 80,000 ALE frames
final_time_step, final_policy_state = init_driver.run()

dataset = replay_buffer.as_dataset(
sample_batch_size=64,
num_steps=2,
num_parallel_calls=3).prefetch(3)

from tf_agents.utils.common import function
collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)

import copy

def train_agent(n_iterations):
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f}".format(
        iteration, train_loss.loss.numpy()), end="")
        ckpt.step.assign_add(1)
        if iteration % 1000 == 0:
            log_metrics(train_metrics)
        if iteration % 1000 == 0:
            save_path = manager.save()
            tf_policy_saver.save(policy_dir)
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
train_agent(n_iterations=200000)

frames = []
def save_frames(trajectory):
    global frames
    frames.append(tf_env.pyenv.envs[0].render(mode="rgb_array"))

watch_driver = DynamicStepDriver(
    tf_env,
    agent.policy,
    observers=[save_frames, ShowProgress(1000)],
    num_steps=1000)
final_time_step, final_policy_state = watch_driver.run()

plot_animation(frames)

import PIL

image_path = os.path.join("images", "rl", "freeway.gif")
frame_images = [PIL.Image.fromarray(frame) for frame in frames[:150]]
frame_images[0].save(image_path, format='GIF',
                     append_images=frame_images[1:],
                     save_all=True,
                     duration=30,
                     loop=0)


# =======================================================================================
# BELOW IS LOADING FROM POLICY AND TRY TO PLAY THE GAME
# =======================================================================================
saved_policy = tf.compat.v2.saved_model.load(policy_dir)
frames = []
def save_frames(trajectory):
    global frames
    frames.append(tf_env.pyenv.envs[0].render(mode="rgb_array"))

watch_driver = DynamicStepDriver(
    tf_env,
    saved_policy,
    observers=[save_frames, ShowProgress(1000)],
    num_steps=1000)
final_time_step, final_policy_state = watch_driver.run()

plot_animation(frames)

import PIL

image_path = os.path.join("images", "rl", "freeway_fromsaved_checkpoint.gif")
frame_images = [PIL.Image.fromarray(frame) for frame in frames[:150]]
frame_images[0].save(image_path, format='GIF',
                     append_images=frame_images[1:],
                     save_all=True,
                     duration=30,
                     loop=0)