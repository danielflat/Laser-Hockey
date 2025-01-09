import optparse
import pickle

import gymnasium as gym
import torch
import torch.nn as nn
from gymnasium import spaces

from src.agents.ddpgagent import ActorCritic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

import torch
import numpy as np


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_fun = torch.nn.Tanh(),
                 output_activation = None):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.output_activation = output_activation
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList([torch.nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations = [activation_fun for l in self.layers]
        self.readout = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)

    def forward(self, x):
        for layer, activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))
        if self.output_activation is not None:
            return self.output_activation(self.readout(x))
        else:
            return self.readout(x)

    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32))).numpy()


# class to store transitions
class Memory():
    def __init__(self, max_size = 100000):
        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size = max_size

    def add_transition(self, transitions_new):
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype = object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        self.transitions[self.current_idx, :] = np.asarray(transitions_new, dtype = object)
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch = 1):
        if batch > self.size:
            batch = self.size
        self.inds = np.random.choice(range(self.size), size = batch, replace = False)
        return self.transitions[self.inds, :]

    def get_all_transitions(self):
        return self.transitions[0:self.size]


class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message = "Unsupported Space"):
        self.message = message
        super().__init__(self.message)


class OUNoise():
    def __init__(self, shape, theta: float = 0.15, dt: float = 1e-2):
        self._shape = shape
        self._theta = theta
        self._dt = dt
        self.noise_prev = np.zeros(self._shape)
        self.reset()

    def __call__(self) -> np.ndarray:
        noise = (
                self.noise_prev
                + self._theta * (- self.noise_prev) * self._dt
                + np.sqrt(self._dt) * np.random.normal(size = self._shape)
        )
        self.noise_prev = noise
        return noise

    def reset(self) -> None:
        self.noise_prev = np.zeros(self._shape)


class DDPGAgent(object):
    """
    Agent implementing Q-learning with NN function approximation.
    """

    def __init__(self, observation_space, action_space, **userconfig):

        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace('Observation space {} incompatible ' \
                                   'with {}. (Require: Box)'.format(observation_space, self))
        if not isinstance(action_space, spaces.box.Box):
            raise UnsupportedSpace('Action space {} incompatible with {}.' \
                                   ' (Require Box)'.format(action_space, self))

        self._observation_space = observation_space
        self._obs_dim = self._observation_space.shape[0]
        self._action_space = action_space
        self._action_n = action_space.shape[0]
        self._config = {
            "eps": 0.1,  # Epsilon: noise strength to add to policy
            "discount": 0.95,
            "buffer_size": int(1e6),
            "batch_size": 128,
            "learning_rate_actor": 0.00001,
            "learning_rate_critic": 0.0001,
            # "hidden_sizes_actor": [128, 128],
            # "hidden_sizes_critic": [128, 128, 64],
            "update_target_every": 100,
            "use_target_net": True
        }
        self._config.update(userconfig)
        self._eps = self._config['eps']

        self.action_noise = OUNoise((self._action_n))

        self.buffer = Memory(max_size = self._config["buffer_size"])

        # # Q Network
        # self.Q = QFunction(observation_dim=self._obs_dim,
        #                    action_dim=self._action_n,
        #                    hidden_sizes= self._config["hidden_sizes_critic"],
        #                    learning_rate = self._config["learning_rate_critic"])
        # # target Q Network
        # self.Q_target = QFunction(observation_dim=self._obs_dim,
        #                           action_dim=self._action_n,
        #                           hidden_sizes= self._config["hidden_sizes_critic"],
        #                           learning_rate = 0)
        #
        # self.policy = Feedforward(input_size=self._obs_dim,
        #                           hidden_sizes= self._config["hidden_sizes_actor"],
        #                           output_size=self._action_n,
        #                           activation_fun = torch.nn.ReLU(),
        #                           output_activation = torch.nn.Tanh())
        # self.policy_target = Feedforward(input_size=self._obs_dim,
        #                                  hidden_sizes= self._config["hidden_sizes_actor"],
        #                                  output_size=self._action_n,
        #                                  activation_fun = torch.nn.ReLU(),
        #                                  output_activation = torch.nn.Tanh())

        self.origin_net = ActorCritic(state_size = self._obs_dim, action_size = self._action_n,
                                      use_compile = False, device = device)
        self.target_net = ActorCritic(state_size = self._obs_dim, action_size = self._action_n,
                                      use_compile = False, device = device)
        self.target_net.eval()  # Set it always to Eval mode
        self.updateTargetNet(source = self.origin_net.actor, target = self.target_net.actor)  # Copy the Q network
        self.updateTargetNet(source = self.origin_net.critic, target = self.target_net.critic)  # Copy the Q network

        # self._copy_nets()

        self.actor_optim = torch.optim.Adam(self.origin_net.actor.parameters(),
                                            lr = 0.00001,
                                            eps = 0.000001)
        self.critic_optim = torch.optim.Adam(self.origin_net.critic.parameters(),
                                             lr = 0.0001,
                                             eps = 0.000001)
        self.criterion = torch.nn.SmoothL1Loss()

        self.USE_BF_16 = False
        self.use_gradient_clipping = False
        self.use_target_net = self._config["use_target_net"]
        self.discount = self._config['discount']
        self.gradient_clipping_value = 1
        self.use_clip_foreach = False
        self.target_net_update_freq = 100

        self.train_iter = 0

    def act(self, observation, eps = None):
        if eps is None:
            eps = self._eps
        #
        action = self.origin_net.greedyAction(
            torch.from_numpy(observation)).detach().numpy() + eps * self.action_noise()  # action in -1 to 1 (+ noise)
        action = self._action_space.low + (action + 1.0) / 2.0 * (self._action_space.high - self._action_space.low)
        return action

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    #
    # def state(self):
    #     return (self.Q.state_dict(), self.policy.state_dict())

    # def restore_state(self, state):
    #     self.Q.load_state_dict(state[0])
    #     self.policy.load_state_dict(state[1])
    #     self._copy_nets()

    def reset(self):
        self.action_noise.reset()

    def train(self, iter_fit = 32):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32))
        losses = []
        self.train_iter += 1
        # if self._config["use_target_net"] and self.train_iter % self._config["update_target_every"] == 0:
        #     # self._copy_nets()
        #     self.updateTargetNet(source = self.origin_net.actor,
        #                          target = self.target_net.actor)
        #     self.updateTargetNet(source = self.origin_net.critic,
        #                          target = self.target_net.critic)
        if self.use_target_net and self.train_iter % self.target_net_update_freq == 0:
            self._copyNets()
            self.updateTargetNet(source = self.origin_net.actor,
                                 target = self.target_net.actor)

            # Step 02: Copy the critic net
            self.updateTargetNet(source = self.origin_net.critic,
                                 target = self.target_net.critic)

        for i in range(iter_fit):
            # sample from the replay buffer
            data = self.buffer.sample(batch = self._config['batch_size'])
            s = to_torch(np.stack(data[:, 0]))  # s_t
            a = to_torch(np.stack(data[:, 1]))  # a_t
            rew = to_torch(np.stack(data[:, 2])[:, None])  # rew  (batchsize,1)
            s_prime = to_torch(np.stack(data[:, 3]))  # s_t+1
            done = to_torch(np.stack(data[:, 4])[:, None])  # done signal  (batchsize,1)

            fit_loss, actor_loss = self.update(s, a, rew, s_prime, done)

            losses.append((fit_loss, actor_loss.item()))

        # if self.train_iter % self.target_net_update_freq == 0:
        #     self._copyNets()
        #
        # if self._config["use_target_net"] and self.train_iter % self._config["update_target_every"] == 0:
        #     # self._copy_nets()
        #     self.updateTargetNet(source = self.origin_net.actor,
        #                          target = self.target_net.actor)
        #     self.updateTargetNet(source = self.origin_net.critic,
        #                          target = self.target_net.critic)

        return losses

    def update(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor,
               done: torch.Tensor):

        # critic update
        if self.USE_BF_16:
            with torch.autocast(device_type = self.device.type, dtype = torch.bfloat16):
                critic_loss = self.critic_forward(action, done, next_state, reward, state)
        else:
            critic_loss = self.critic_forward(action, done, next_state, reward, state)

        # critic backward step
        self.critic_optim.zero_grad()
        critic_loss.backward()
        if self.use_gradient_clipping:
            torch.nn.utils.clip_grad_value_(parameters = self.origin_net.critic.parameters(),
                                            clip_value = self.gradient_clipping_value,
                                            foreach = self.use_clip_foreach)
        self.critic_optim.step()

        if self.USE_BF_16:
            with torch.autocast(device_type = self.device.type, dtype = torch.bfloat16):
                actor_loss = self.actor_forward(state)
        else:
            actor_loss = self.actor_forward(state)

        # Step 05: Optimize the actor net
        self.actor_optim.zero_grad()
        # actor backward step
        actor_loss.backward()
        if self.use_gradient_clipping:
            torch.nn.utils.clip_grad_value_(parameters = self.origin_net.actor.parameters(),
                                            clip_value = self.gradient_clipping_value,
                                            foreach = self.use_clip_foreach)
        self.actor_optim.step()

        # NOTE: For logging, we currently only consider the critic loss
        return critic_loss, actor_loss

    def actor_forward(self, state):
        # actor forward step: Maximize the actor network by go and maximize the critic network
        q_greedy = self.origin_net.QValue(state = state, action = self.origin_net.greedyAction(state))
        actor_loss = -torch.mean(q_greedy)
        return actor_loss

    def critic_forward(self, action, done, next_state, reward, state):
        # Step 01: Compute the td target
        if self.use_target_net:
            q_target = self.target_net.QValue(state = next_state, action = self.target_net.greedyAction(next_state))
        else:
            q_target = self.origin_net.QValue(state = next_state, action = self.origin_net.greedyAction(next_state))
        td_target = reward + (1 - done) * self.discount * q_target.detach()
        # Step 02: Compute the prediction
        q_origin = self.origin_net.QValue(state, action)
        # Step 03: critic loss
        critic_loss = self.criterion(q_origin, td_target)
        return critic_loss

    def updateTargetNet(self, source: nn.Module, target: nn.Module) -> None:
        """
        Updates the target network with the weights of the original one
        """
        # Do a hard parameter update. Copy all values from the origin to the target network
        target.load_state_dict(source.state_dict())

    def _copyNets(self):
        # Step 01: Copy the actor net
        self.updateTargetNet(source = self.origin_net.actor,
                             target = self.target_net.actor)

        # Step 02: Copy the critic net
        self.updateTargetNet(source = self.origin_net.critic,
                             target = self.target_net.critic)


def main():
    optParser = optparse.OptionParser()
    optParser.add_option('-e', '--env', action = 'store', type = 'string',
                         dest = 'env_name', default = "Pendulum-v1",
                         help = 'Environment (default %default)')
    optParser.add_option('-n', '--eps', action = 'store', type = 'float',
                         dest = 'eps', default = 0.1,
                         help = 'Policy noise (default %default)')
    optParser.add_option('-t', '--train', action = 'store', type = 'int',
                         dest = 'train', default = 32,
                         help = 'number of training batches per episode (default %default)')
    optParser.add_option('-l', '--lr', action = 'store', type = 'float',
                         dest = 'lr', default = 0.0001,
                         help = 'learning rate for actor/policy (default %default)')
    optParser.add_option('-m', '--maxepisodes', action = 'store', type = 'float',
                         dest = 'max_episodes', default = 2000,
                         help = 'number of episodes (default %default)')
    optParser.add_option('-u', '--update', action = 'store', type = 'float',
                         dest = 'update_every', default = 100,
                         help = 'number of episodes between target network updates (default %default)')
    optParser.add_option('-s', '--seed', action = 'store', type = 'int',
                         dest = 'seed', default = None,
                         help = 'random seed (default %default)')
    opts, args = optParser.parse_args()
    ############## Hyperparameters ##############
    env_name = opts.env_name
    # creating environment
    if env_name == "LunarLander-v2":
        env = gym.make(env_name, continuous = True)
    else:
        env = gym.make(env_name)
    render = False
    log_interval = 20  # print avg reward in the interval
    max_episodes = opts.max_episodes  # max training episodes
    max_timesteps = 2000  # max timesteps in one episode

    train_iter = opts.train  # update networks for given batched after every episode
    eps = opts.eps  # noise of DDPG policy
    lr = opts.lr  # learning rate of DDPG policy
    random_seed = opts.seed
    #############################################

    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    ddpg = DDPGAgent(env.observation_space, env.action_space, eps = eps, learning_rate_actor = lr,
                     update_target_every = opts.update_every)

    # logging variables
    rewards = []
    lengths = []
    losses = []
    timestep = 0

    def save_statistics():
        with open(f"./results/DDPG_{env_name}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}-stat.pkl", 'wb') as f:
            pickle.dump({"rewards": rewards, "lengths": lengths, "eps": eps, "train": train_iter,
                         "lr": lr, "update_every": opts.update_every, "losses": losses}, f)

    # training loop
    for i_episode in range(1, max_episodes + 1):
        ob, _info = env.reset()
        ddpg.reset()
        total_reward = 0
        for t in range(max_timesteps):
            timestep += 1
            done = False
            a = ddpg.act(ob)
            (ob_new, reward, done, trunc, _info) = env.step(a)
            total_reward += reward
            ddpg.store_transition((ob, a, reward, ob_new, done))
            ob = ob_new
            if done or trunc: break

        losses.extend(ddpg.train(train_iter))

        rewards.append(total_reward)
        lengths.append(t)

        # save every 500 episodes
        # if i_episode % 500 == 0:
        #     print("########## Saving a checkpoint... ##########")
        #     torch.save(ddpg.state(), f'./results/DDPG_{env_name}_{i_episode}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}.pth')
        #     save_statistics()

        # logging
        if i_episode % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            avg_length = int(np.mean(lengths[-log_interval:]))

            print(f"Episode {i_episode} \t avg length: {avg_length} \t reward: {avg_reward}, ")
    # save_statistics()


if __name__ == '__main__':
    main()
