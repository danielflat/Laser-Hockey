import torch
from torch import device, nn
from torch.distributions import Categorical

from src.agent import Agent
from src.replaymemory import ReplayMemory


class ActorCritic(nn.Module):
    def __init__(self, input_shape: int, action_shape: int):
        super().__init__()
        # It yields the prob. distribution of the action space given the state the given state
        self.actor = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, action_shape),
            nn.Softmax(dim=-1))

        # It yields the state function for the given state
        self.critic = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1))

    def flow_actor(self, x: torch.Tensor):
        x = self.actor(x)
        return x

    def flow_critic(self, x: torch.Tensor):
        x = self.critic(x)
        return x


class PPOAgent(Agent):
    def __init__(self, observation_size, action_size, hyperparams: dict, options: dict, optim: dict, device: device):
        super().__init__()

        # Hyperparams
        self.opt_iter = hyperparams["OPT_ITER"]
        self.batch_size = hyperparams["BATCH_SIZE"]
        self.discount = hyperparams["DISCOUNT"]
        self.epsilon = hyperparams["EPSILON"]
        self.epsilon_start = hyperparams["EPSILON"]
        self.epsilon_min = hyperparams["EPSILON_MIN"]
        self.epsilon_decay = hyperparams["EPSILON_DECAY"]
        self.gradient_clipping_value = hyperparams["GRADIENT_CLIPPING_VALUE"]
        self.target_net_update_freq = hyperparams["TARGET_NET_UPDATE_FREQ"]
        self.tau = hyperparams["TAU"]
        self.eps_clip = hyperparams["EPS_CLIP"]

        # Options
        self.use_target_net = options["USE_TARGET_NET"]
        self.use_soft_updates = options["USE_SOFT_UPDATES"]
        self.use_gradient_clipping = options["USE_GRADIENT_CLIPPING"]
        self.epsilon_decay_strategy = options["EPSILON_DECAY_STRATEGY"]
        self.device: device = device
        self.use_clip_foreach = options["USE_CLIP_FOREACH"]
        self.USE_BF_16 = options["USE_BF16"]

        self.policy_net = ActorCritic(input_shape=observation_size, action_shape=action_size)
        self.policy_net.to(device)

        self.old_policy_net = ActorCritic(input_shape=observation_size, action_shape=action_size)
        self.old_policy_net.to(device)

        # set old policy net always to eval mode
        self.old_policy_net.eval()

        # copy the network
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = self.initOptim(optim=optim, parameters=self.policy_net.parameters())

        self.criterion = nn.MSELoss()

    def optimize(self, memory: ReplayMemory, episode_i: int) -> list[float]:
        """
                This function is used to train and optimize the Q Network with the help of the replay memory.
                :return: A list of all losses during optimization
                """
        assert self.isEval == False, "Make sure to put the agent in training mode before calling the opt. routine"

        losses = []

        # Since we do Monte Carlo Estimation, we sample the whole trajectory of the episode
        batch_size = len(memory)
        state, action, reward, next_state, done, info = memory.sample(batch_size, randomly=False)

        # Next, we have to discount the reward w.t.r. the discount factor
        exponents = torch.arange(len(reward), dtype=torch.float32)
        discount_factors = torch.pow(self.discount, exponents)[:, None]
        reward = reward * discount_factors

        # Now, lets squeeze some inputs
        action = action.squeeze(1)
        reward = reward.squeeze(1)

        # We start at i=1 to prevent a direct update of the weights
        for i in range(1, self.opt_iter + 1):
            self.optimizer.zero_grad()

            # Forward step
            if self.USE_BF_16:
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    loss = self.forward_pass(state, action, reward, next_state, done)
            else:
                loss = self.forward_pass(state, action, reward, next_state, done)

            # Track the loss
            losses.append(loss.item())

            # Backward step
            loss.backward()
            # if we want to clip our gradients
            if self.use_gradient_clipping:
                # In-place gradient clipping
                torch.nn.utils.clip_grad_value_(parameters=self.policy_net.parameters(),
                                                clip_value=self.gradient_clipping_value,
                                                foreach=self.use_clip_foreach)
            self.optimizer.step()

            # Update the target net after some iterations again
            if self.use_target_net and i % self.target_net_update_freq == 0:
                self.updateOldPolicyNet(soft_update=self.use_soft_updates)

        # in PPO, we have to clear the memory after each optimization loop, since
        memory.clear()

        return losses

    def act(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            # if you
            if self.isEval:
                # TODO
                action_probs = self.policy_net.flow_actor(state)
                categorical = Categorical(action_probs)
                action = categorical.sample()

                return action.item()
            else:
                # take the actions w.r.t. the old policy
                action_probs = self.old_policy_net.flow_actor(state)
                categorical = Categorical(action_probs)
                action = categorical.sample()

                return action.item()

    def setMode(self, eval=False) -> None:
        """
        Set the Agent in training or evaluation mode
        :param eval: If true = eval mode, False = training mode
        """
        self.isEval = eval
        if self.isEval:
            self.policy_net.eval()
        else:
            self.policy_net.train()

    def forward_pass(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor,
                     done: torch.Tensor) -> torch.Tensor:
        # Step 01: First, we need the logprobs of the old policy
        old_log_probs = self.logprobs(self.old_policy_net, state)

        # Step 02: Evaluate the actions
        logprobs, state_values, dist_entropy = self.evaluate(state, action)

        # Step 03: Compute the ratio
        ratios = torch.exp(logprobs - old_log_probs.detach())

        # Step 04: Compute the advantages
        advantages = reward - state_values.detach()

        # Step 05: Compute the surrogate loss
        objective = ratios * advantages
        objective_clipped = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        # Step 06: Finally the loss
        loss = -torch.min(objective, objective_clipped) + 0.5 * self.criterion(state_values,
                                                                               reward) - 0.01 * dist_entropy

        return loss.mean()

    def logprobs(self, net, state):
        action_probs = net.flow_actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        return dist.log_prob(action)

    def evaluate(self, state, action):
        action_probs = self.policy_net.flow_actor(state)
        dist = Categorical(action_probs)

        # Step 01: First, get the log prob of the action given the state
        action_logprobs = dist.log_prob(action)

        # Step 02: Next, we get the entropy of the action space
        dist_entropy = dist.entropy()

        # Step 03: Finally, we acquire the value of the state.
        # We squeeze it to have (batch_size,) shape
        state_value = self.policy_net.flow_critic(state).squeeze(1)

        return action_logprobs, state_value, dist_entropy

    # TODO:
    def saveModel(self, model_name: str, iteration: int) -> None:
        pass

    def loadModel(self, file_name: str) -> None:
        pass

    def updateOldPolicyNet(self, soft_update: bool):
        """
        Updates the target network with the weights of the original one
        """
        assert self.use_target_net == True, "You must use have 'self.use_target == True' to call 'updateTargetNet()'"

        if soft_update:
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′ where θ′ are the target net weights
            target_net_state_dict = self.old_policy_net.state_dict()
            origin_net_state_dict = self.policy_net.state_dict()
            for key in origin_net_state_dict:
                target_net_state_dict[key] = origin_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                        1 - self.tau)
            self.old_policy_net.load_state_dict(target_net_state_dict)
        else:
            # Do a hard parameter update. Copy all values from the origin to the target network
            self.old_policy_net.load_state_dict(self.policy_net.state_dict())
