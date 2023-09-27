import numpy as np
import gym
import random
import matplotlib.pyplot as plt
RENDER = False
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.normal import Normal
import os
from torch.distributions import Categorical
import torch.optim as optim
import torch.nn.functional as F
from arguments import args

eps = np.finfo(np.float32).eps.item()

all_rewards = []
running_rewards = []


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# 利用当前策略进行采样，产生数据
class Sample():
    def __init__(self, env):
        self.env = env
        self.gamma = 0.99
        self.last_state = None
        self.batch_state = None
        self.batch_act = None
        self.batch_logp = None
        self.batch_val_target = None
        self.index = None
        self.episode_return = 0
        self.sum_return = 0

    # 采样轨迹
    def sample_episode(self, actor_net, num_episodes):
        Flag = 0
        val_target = 0
        episode_state = []
        episode_vals = []
        episode_actions = []
        episode_val_target = []
        done = False
        episode_sum = 0
        log_probs = []
        for episode in range(num_episodes):
            state = self.env.reset()
            steps = 0
            running_add = 0
            rewards = []
            while True:
                episode_state.append(state)
                action, log_prob = actor_net.get_a(state)
                log_probs.append(log_prob)
                episode_actions.append(action)
                episode_val_target.append(0.0)
                next_state, reward, done, _ = self.env.step(action)
                # self.env.render()
                state = next_state
                rewards.append(reward)
                if steps >= 200:
                    break
                steps = steps + 1
                if done:
                    self.last_state = next_state
                    val_target = 0.0
                    episode_vals.append(val_target)
                    for t in reversed(range(len(rewards))):
                        running_add = running_add * self.gamma + rewards[t]
                        episode_val_target[t] = running_add
                        episode_sum += rewards[t]
                    all_rewards.append(np.sum(rewards))
                    running_rewards.append(np.mean(all_rewards[-30:]))
                    episode_val_target = (episode_val_target - np.mean(episode_val_target)) / (
                            np.std(episode_val_target) + eps)
                    break

            # episode_state = np.reshape(episode_state, [len(episode_state), actor_net.n_features])
            # episode_actions = np.reshape(episode_actions, [len(episode_actions),1])
            # episode_val_target = np.reshape(episode_val_target, [len(episode_val_target),1])
            # episode_state = torch.as_tensor(episode_state, dtype=torch.float32)
            # episode_actions = torch.as_tensor(episode_actions, dtype=torch.float32)
            # episode_val_target = torch.as_tensor(episode_val_target, dtype=torch.float32)
        return episode_state, episode_actions, episode_val_target, all_rewards, running_rewards, log_probs

    # 最终测试网络
    def test_sample_one_episode(self, actor_net):
        episode_state = []
        episode_vals = []
        episode_actions = []
        episode_rewards = []
        episode_val_target = []
        done = False
        num_episodes = 1
        episode_sum = 0
        for i in range(num_episodes):
            state = self.env.reset()
            steps = 0
            while True:
                episode_state.append(state)
                # 采样动作，及动作的对数，不计算梯度
                action, log_a = actor_net.get_a(state)
                episode_actions.append(action)
                episode_val_target.append(0.0)
                # 往前推进一步
                next_state, reward, done, _ = self.env.step(action)
                # 渲染环境
                self.env.render()
                state = next_state
                episode_rewards.append(reward)
                steps = steps + 1
                if steps>=1000:
                    break
                # 处理回报
                if (done):
                    self.last_state = next_state
                    val_target = 0.0
                    episode_vals.append(val_target)
                    discounted_sum_reward = np.zeros_like(episode_rewards)
                    # 计算mbatch折扣累积回报
                    # print("总长度：",len(episode_rewards))
                    for t in reversed(range(0, len(episode_rewards))):
                        val_target = episode_rewards[t] + val_target * self.gamma
                        episode_val_target[t] = val_target
                        episode_sum += episode_rewards[t]
                    break


# 网络结构
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=24, learning_rate=0.01):
        super(PolicyNetwork, self).__init__()

        self.n_features = num_inputs
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = F.softmax(self.fc2(x), dim=1)
        return x

    # 采样一个动作
    def get_a(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        act_probs = self.forward(state)
        c = Categorical(act_probs)
        action = c.sample()
        return action.item(), c.log_prob(action)

    # 更新策略网络结构
    def update_policy(self, vts, log_probs):
        policy_loss = []
        for log_prob, vt in zip(log_probs, vts):
            policy_loss.append(-log_prob * vt)

        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

    def discounted_norm_rewards(self, rewards, GAMMA):
        vt = np.zeros_like(rewards)
        running_add = 0
        episode_sum = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * GAMMA + rewards[t]
            vt[t] = running_add
            episode_sum += rewards[t]
        # normalized discounted rewards
        vt = (vt - np.mean(vt)) / (np.std(vt) + eps)
        return vt, episode_sum


# Policy_Gradient算法类
class Policy_Gradient():
    def __init__(self, actor, env, lr=0.01):
        # 1. 定义网络模型
        self.actor_net = actor
        self.optimizer = optim.Adam(self.actor_net.parameters(), lr=lr)
        self.env = env

    # 训练
    def pg_train(self, training_num):
        reward_sum = 0.0
        reward_sum_line = []
        training_time = []
        for i in range(training_num):
            sampler = Sample(self.env)
            train_state, train_actions, train_vt, rewards, running_rewards, train_log = sampler.sample_episode(
                self.actor_net, 1)
            self.actor_net.update_policy(train_vt, train_log)
            print('episode:', i, 'total reward: ', rewards[-1], 'running reward:', int(running_rewards[-1]))
        torch.save(self.actor_net, "policy_Net.pkl")



def test():
    actor_net = torch.load('policy_net.pkl')
    sampler = Sample(env)
    sampler.test_sample_one_episode(actor_net)


def train(training_num):
    actor_net = PolicyNetwork(state_space, action_space)
    pg = Policy_Gradient(actor_net, env, lr=0.003)
    pg.pg_train(training_num)


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    env = env.unwrapped
    env.seed(1)

    np.random.seed(1)
    torch.manual_seed(1)
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    if args.run_mode=='train':
        train(args.train_num)
    
    elif args.run_mode=='test':
        test()
