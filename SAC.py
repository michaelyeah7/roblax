import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution

import math

from numpy.random import default_rng
from torch.distributions import constraints
from torch.distributions.transforms import Transform
from torch.nn.functional import softplus

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class ReplayPool:

    def __init__(self, action_dim, state_dim, capacity=1e6):
        self.capacity = int(capacity)
        self._action_dim = action_dim
        self._state_dim = state_dim
        self._pointer = 0
        self._size = 0
        self._init_memory()
        self._rng = default_rng()

    def _init_memory(self):
        self._memory = {
            'state': np.zeros((self.capacity, self._state_dim), dtype='float32'),
            'action': np.zeros((self.capacity, self._action_dim), dtype='float32'),
            'reward': np.zeros((self.capacity), dtype='float32'),
            'nextstate': np.zeros((self.capacity, self._state_dim), dtype='float32'),
            'real_done': np.zeros((self.capacity), dtype='bool')
        }

    def push(self, transition: Transition):

        # Handle 1-D Data
        num_samples = transition.state.shape[0] if len(transition.state.shape) > 1 else 1
        idx = np.arange(self._pointer, self._pointer + num_samples) % self.capacity

        for key, value in transition._asdict().items():
            self._memory[key][idx] = value

        self._pointer = (self._pointer + num_samples) % self.capacity
        self._size = min(self._size + num_samples, self.capacity)

    def _return_from_idx(self, idx):
        sample = {k: tuple(v[idx]) for k,v in self._memory.items()}
        return Transition(**sample)

    def sample(self, batch_size: int, unique: bool = True):
        idx = np.random.randint(0, self._size, batch_size) if not unique else self._rng.choice(self._size, size=batch_size, replace=False)
        return self._return_from_idx(idx)

    def sample_all(self):
        return self._return_from_idx(np.arange(0, self._size))

    def __len__(self):
        return self._size

    def clear_pool(self):
        self._init_memory()

    def initialise(self, old_pool):
        # Not Tested
        old_memory = old_pool.sample_all()
        for key in self._memory:
            self._memory[key] = np.append(self._memory[key], old_memory[key], 0)


# Taken from: https://github.com/pytorch/pytorch/pull/19785/files
# The composition of affine + sigmoid + affine transforms is unstable numerically
# tanh transform is (2 * sigmoid(2x) - 1)
# Old Code Below:
# transforms = [AffineTransform(loc=0, scale=2), SigmoidTransform(), AffineTransform(loc=-1, scale=2)]
class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2. * (math.log(2.) - x - softplus(-2. * x))

class MLPNetwork(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
                        nn.Linear(input_dim, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, output_dim),
                        )
    
    def forward(self, x):
        return self.network(x)


class Policy(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Policy, self).__init__()
        self.action_dim = action_dim
        self.network = MLPNetwork(state_dim, action_dim * 2, hidden_size)

    def forward(self, x, get_logprob=False):
        mu_logstd = self.network(x)
        mu, logstd = mu_logstd.chunk(2, dim=1)
        logstd = torch.clamp(logstd, -20, 2)
        std = logstd.exp()
        dist = Normal(mu, std)
        transforms = [TanhTransform(cache_size=1)]
        dist = TransformedDistribution(dist, transforms)
        action = dist.rsample()
        if get_logprob:
            logprob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        else:
            logprob = None
        mean = torch.tanh(mu)
        return action, logprob, mean


class DoubleQFunc(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DoubleQFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)


class SAC:

    def __init__(self, seed, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=5e-3, batchsize=256, hidden_size=256, update_interval=1, buffer_size=int(1e6), target_entropy=None):
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy if target_entropy else -action_dim
        self.batchsize = batchsize
        self.update_interval = update_interval

        torch.manual_seed(seed)

        # aka critic
        self.q_funcs = DoubleQFunc(state_dim, action_dim, hidden_size=hidden_size).to(device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # aka actor
        self.policy = Policy(state_dim, action_dim, hidden_size=hidden_size).to(device)

        # aka temperature
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.replay_pool = ReplayPool(action_dim=action_dim, state_dim=state_dim, capacity=int(1e6))
    
    def get_action(self, state, state_filter=None, deterministic=False):
        if state_filter:
            state = state_filter(state)
        with torch.no_grad():
            action, _, mean = self.policy(torch.Tensor(state).view(1,-1).to(device))
        if deterministic:
            return mean.squeeze().cpu().numpy()
        return np.atleast_1d(action.squeeze().cpu().numpy())

    def update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def update_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, done_batch):
        with torch.no_grad():
            nextaction_batch, logprobs_batch, _ = self.policy(nextstate_batch, get_logprob=True)
            q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
            # take min to mitigate positive bias in q-function training
            q_target = torch.min(q_t1, q_t2)
            value_target = reward_batch + (1.0 - done_batch) * self.gamma * (q_target - self.alpha * logprobs_batch)
        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        loss_1 = F.mse_loss(q_1, value_target)
        loss_2 = F.mse_loss(q_2, value_target)
        return loss_1, loss_2

    def update_policy_and_temp(self, state_batch):
        action_batch, logprobs_batch, _ = self.policy(state_batch, get_logprob=True)
        q_b1, q_b2 = self.q_funcs(state_batch, action_batch)
        qval_batch = torch.min(q_b1, q_b2)
        policy_loss = (self.alpha * logprobs_batch - qval_batch).mean()
        temp_loss = -self.alpha * (logprobs_batch.detach() + self.target_entropy).mean()
        return policy_loss, temp_loss

    def optimize(self, n_updates, state_filter=None):
        q1_loss, q2_loss, pi_loss, a_loss = 0, 0, 0, 0
        for i in range(n_updates):
            samples = self.replay_pool.sample(self.batchsize)

            if state_filter:
                state_batch = torch.FloatTensor(state_filter(samples.state)).to(device)
                nextstate_batch = torch.FloatTensor(state_filter(samples.nextstate)).to(device)
            else:
                state_batch = torch.FloatTensor(samples.state).to(device)
                nextstate_batch = torch.FloatTensor(samples.nextstate).to(device)
            action_batch = torch.FloatTensor(samples.action).to(device)
            reward_batch = torch.FloatTensor(samples.reward).to(device).unsqueeze(1)
            done_batch = torch.FloatTensor(samples.real_done).to(device).unsqueeze(1)
            
            # update q-funcs
            q1_loss_step, q2_loss_step = self.update_q_functions(state_batch, action_batch, reward_batch, nextstate_batch, done_batch)
            q_loss_step = q1_loss_step + q2_loss_step
            self.q_optimizer.zero_grad()
            q_loss_step.backward()
            self.q_optimizer.step()

            # update policy and temperature parameter
            for p in self.q_funcs.parameters():
                p.requires_grad = False
            pi_loss_step, a_loss_step = self.update_policy_and_temp(state_batch)
            self.policy_optimizer.zero_grad()
            pi_loss_step.backward()
            self.policy_optimizer.step()
            self.temp_optimizer.zero_grad()
            a_loss_step.backward()
            self.temp_optimizer.step()
            for p in self.q_funcs.parameters():
                p.requires_grad = True

            q1_loss += q1_loss_step.detach().item()
            q2_loss += q2_loss_step.detach().item()
            pi_loss += pi_loss_step.detach().item()
            a_loss += a_loss_step.detach().item()
            if i % self.update_interval == 0:
                self.update_target()
        return q1_loss, q2_loss, pi_loss, a_loss

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path+'_q1')
        torch.save(self.soft_q_net2.state_dict(), path+'_q2')
        torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path+'_q1'))
        self.soft_q_net2.load_state_dict(torch.load(path+'_q2'))
        self.policy_net.load_state_dict(torch.load(path+'_policy'))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()
