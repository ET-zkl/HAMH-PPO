from . import RLAgent
from common.registry import Registry
import numpy as np
import os
import random
from collections import OrderedDict, deque
import gym

from generator.lane_vehicle import LaneVehicleGenerator
from generator.intersection_phase import IntersectionPhaseGenerator
import torch
from torch import nn
import torch.nn.functional as F
import torch_scatter
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops
import itertools


@Registry.register_model('ippo')
class IPPO(RLAgent):
    #  TODO: test multiprocessing effect on agents or need deep copy here
    def __init__(self, world, rank):
        super().__init__(world, world.intersection_ids[rank])
        """
        multi-agents in one model-> modify self.action_space, self.reward_generator, self.ob_generator here
        """
        #  general setting of world and model structure
        # TODO: different phases matching
        #self.buffer_size = Registry.mapping['trainer_mapping']['trainer_setting'].param['buffer_size']
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.device = torch.device("cuda",3) if torch.cuda.is_available() else torch.device("cpu")

        self.graph = Registry.mapping['world_mapping']['graph_setting'].graph
        #self.device = torch.device("cpu")
        print('self.device', self.device)
        print('训练一个智能体共享参数--------------')
        # print(Registry.mapping['world_mapping']['graph_setting'])
        self.world = world
        self.sub_agents = len(self.world.intersections)  # 3
        # TODO: support dynamic graph later  边的信息
        self.edge_idx = torch.tensor(self.graph['sparse_adj'].T, dtype=torch.long).to(self.device)  # source -> target

        #  model parameters
        self.phase = Registry.mapping['model_mapping']['setting'].param['phase']
        self.one_hot = Registry.mapping['model_mapping']['setting'].param['one_hot']
        self.model_dict = Registry.mapping['model_mapping']['setting'].param

        #  get generator for CoLightAgent
        observation_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]  # id：名字；idx：索引
            tmp_generator = LaneVehicleGenerator(self.world, inter, ['lane_count'], in_only=True, average=None)
            # tmp_generator：一个路口四条路，每条路三个车道
            observation_generators.append((node_idx, tmp_generator))
        sorted(observation_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.ob_generator = observation_generators

        #  get reward generator for CoLightAgent
        rewarding_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"],
                                                 in_only=True, average='all', negative=True)
            rewarding_generators.append((node_idx, tmp_generator))
        sorted(rewarding_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.reward_generator = rewarding_generators

        #  get queue generator for CoLightAgent
        queues = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"],
                                                 in_only=True, negative=False)
            queues.append((node_idx, tmp_generator))
        # now generator's order is according to its index in graph
        sorted(queues, key=lambda x: x[0])
        self.queue = queues

        #  get delay generator for CoLightAgent
        delays = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_delay"],
                                                 in_only=True, average="all", negative=False)
            delays.append((node_idx, tmp_generator))
        # now generator's order is according to its index in graph
        sorted(delays, key=lambda x: x[0])
        self.delay = delays

        #  phase generator
        phasing_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = IntersectionPhaseGenerator(self.world, inter, ['phase'],
                                                       targets=['cur_phase'], negative=False)
            phasing_generators.append((node_idx, tmp_generator))
        sorted(phasing_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.phase_generator = phasing_generators

        # TODO: add irregular control of signals in the future
        self.action_space = gym.spaces.Discrete(len(self.world.intersections[0].phases))  # self.world.intersections[0].phases = [1,2,3,4,5,6,7,8]
        print("==============================")
        print(self.ob_generator[0][1])
        if self.phase:
            # TODO: irregular ob and phase in the future
            if self.one_hot:
                self.ob_length = self.ob_generator[0][1].ob_length + len(self.world.intersections[0].phases)  # 12 + 8 车道数+相位数
            else:
                self.ob_length = self.ob_generator[0][1].ob_length + 1
        else:
            self.ob_length = self.ob_generator[0][1].ob_length  # 12

        self.get_attention = Registry.mapping['logger_mapping']['setting'].param['get_attention']
        # train parameters
        self.rank = rank  # 0
        self.gamma = Registry.mapping['model_mapping']['setting'].param['gamma']
        self.grad_clip = Registry.mapping['model_mapping']['setting'].param['grad_clip']
        self.epsilon = Registry.mapping['model_mapping']['setting'].param['epsilon']
        self.epsilon_decay = Registry.mapping['model_mapping']['setting'].param['epsilon_decay']
        self.epsilon_min = Registry.mapping['model_mapping']['setting'].param['epsilon_min']
        self.learning_rate = Registry.mapping['model_mapping']['setting'].param['learning_rate']
        self.vehicle_max = Registry.mapping['model_mapping']['setting'].param['vehicle_max']
        self.batch_size = Registry.mapping['model_mapping']['setting'].param['batch_size']
        self.c_learning_rate = Registry.mapping['model_mapping']['setting'].param['c_learning_rate']
        self.a_learning_rate = Registry.mapping['model_mapping']['setting'].param['a_learning_rate']
        self.eps = 0.2
        self.lmbda = 0.95
        self.rnn_hidden_dim = 128

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        self.actor = Actor(self.ob_length, self.action_space.n).to(self.device)
        self.criterion = nn.MSELoss(reduction='mean')
        self.critic_optimizer = torch.optim.Adam(itertools.chain(self.model.parameters()), lr=self.c_learning_rate, eps=1e-5)
        self.actor_optimizer = torch.optim.Adam(itertools.chain(self.actor.parameters()), lr=self.a_learning_rate, eps=1e-5)

        # self.optimizer = optim.RMSprop(self.model.parameters(),
        #                                lr=self.learning_rate,
        #                                alpha=0.9, centered=False, eps=1e-7)
    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.hidden_state = torch.zeros((episode_num, self.sub_agents, self.rnn_hidden_dim)).to(self.device)  # cuda
    def reset(self):
        observation_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]  # 'intersection_1_1'
            node_idx = self.graph['node_id2idx'][node_id]  # 0 1
            tmp_generator = LaneVehicleGenerator(self.world, inter, ['lane_count'], in_only=True, average=None)
            observation_generators.append((node_idx, tmp_generator))
        sorted(observation_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.ob_generator = observation_generators
        # print("==============================")
        # print(self.ob_generator[0][1])
        #  get reward generator for CoLightAgent
        rewarding_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"],
                                                 in_only=True, average='all', negative=True)
            rewarding_generators.append((node_idx, tmp_generator))
        sorted(rewarding_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.reward_generator = rewarding_generators

        #  phase generator
        phasing_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = IntersectionPhaseGenerator(self.world, inter, ['phase'],
                                                       targets=['cur_phase'], negative=False)
            phasing_generators.append((node_idx, tmp_generator))
        sorted(phasing_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.phase_generator = phasing_generators

        # queue metric
        queues = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"],
                                                 in_only=True, negative=False)
            queues.append((node_idx, tmp_generator))
        # now generator's order is according to its index in graph
        sorted(queues, key=lambda x: x[0])
        self.queue = queues

        # delay metric
        delays = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_delay"],
                                                 in_only=True, average="all", negative=False)
            delays.append((node_idx, tmp_generator))
        # now generator's order is according to its index in graph
        sorted(delays, key=lambda x: x[0])
        self.delay = delays

    def get_ob(self):
        x_obs = []  # sub_agents * lane_nums,
        for i in range(len(self.ob_generator)):
            x_obs.append((self.ob_generator[i][1].generate()) / self.vehicle_max)
        # construct edge information.
        length = set([len(i) for i in x_obs])
        if len(length) == 1: # each intersections may has  different lane nums
            x_obs = np.array(x_obs, dtype=np.float32)
        else:
            x_obs = [np.expand_dims(x,axis=0) for x in x_obs]
        return x_obs

    def get_reward(self):
        # TODO: test output
        rewards = []  # sub_agents
        for i in range(len(self.reward_generator)):
            rewards.append(self.reward_generator[i][1].generate())
        rewards = np.squeeze(np.array(rewards)) * 12
        return rewards

    def get_phase(self):
        # TODO: test phase output onehot/int
        phase = []  # sub_agents
        for i in range(len(self.phase_generator)):
            phase.append((self.phase_generator[i][1].generate()))  #[[0],[0],[0]]
        phase = (np.concatenate(phase)).astype(np.int8)
        # phase = np.concatenate(phase, dtype=np.int8)
        return phase  # [0 0 0]

    def get_queue(self):
        """
        get delay of intersection
        return: value(one intersection) or [intersections,](multiple intersections)
        """
        queue = []
        for i in range(len(self.queue)):
            queue.append((self.queue[i][1].generate()))
        tmp_queue = np.squeeze(np.array(queue))
        queue = np.sum(tmp_queue, axis=1 if len(tmp_queue.shape)==2 else 0)
        return queue

    def get_delay(self):
        delay = []
        for i in range(len(self.delay)):
            delay.append((self.delay[i][1].generate()))
        delay = np.squeeze(np.array(delay))
        return delay # [intersections,]

    def get_action(self, ob, rnn_state, test=False):
        """
        input are np.array here
        # TODO: support irregular input in the future
        :param ob: [agents, ob_length] -> [batch, agents, ob_length]
        :param phase: [agents] -> [batch, agents]
        :param test: boolean, exploit while training and determined while testing
        :return: [batch, agents] -> action taken by environment
        """
        # if not test:
        #     if np.random.rand() <= self.epsilon:
        #         return self.sample()
        observation = torch.tensor(ob, dtype=torch.float32).to(self.device)
        edge = self.edge_idx
        dp = Data(x=observation, edge_index=edge)
        # TODO: not phase not used
        if rnn_state is not None:
            rnn_state = rnn_state.to(self.device)  # cuda
        action_dist, rnn_state = self.actor(observation, rnn_state)
        action = action_dist.sample()
        action_log_probs = action_dist.log_prob(action).to(self.device)
        return action.view(-1).cpu().clone().numpy(), action_log_probs, rnn_state, None
        # if self.get_attention:
        #     # TODO: collect attention matrix later
        #     actions = self.model(x=dp.x, edge_index=dp.edge_index, train=False)
        #     att = None
        #     actions = actions.clone().detach().numpy()
        #     return np.argmax(actions, axis=1), att  # [batch, agents], [batch, agents, nv, neighbor]
        # else:
        #     actions = self.model(x=dp.x, edge_index=dp.edge_index, train=False)
        #     actions = actions.clone().detach().numpy()
        #     a = np.argmax(actions, axis=1)
        #     b = self.model.module_list
        #     return np.argmax(actions, axis=1)  # [batch, agents] TODO: check here

    def sample(self):
        return np.random.randint(0, self.action_space.n, self.sub_agents)  #

    def _build_model(self):
        model = ColightNet(self.ob_length, self.action_space.n, **self.model_dict).to(self.device)
        return model

    def remember(self, last_obs, last_phase, actions, rewards, obs, cur_phase, done, key):
        self.replay_buffer.append((key, (last_obs, last_phase, actions, rewards, obs, cur_phase)))

    #def remember(self, last_obs, last_phase, actions, rewards, obs, cur_phase, key):
    #    self.replay_buffer.append((key, (last_obs, last_phase, actions, rewards, obs, cur_phase)))

    def _batchwise(self, samples):
        # load onto tensor

        batch_list = []
        batch_list_p = []
        actions = []
        rewards = []
        act_log_prob = torch.empty((0), dtype=torch.float32).to(self.device)
        for item in samples:
            dp = item[1]
            act_log_prob = torch.cat((act_log_prob, dp[4]), 0)
            state = torch.tensor(dp[0], dtype=torch.float32).to(self.device)  # 获取观测信息
            batch_list.append(Data(x=state, edge_index=self.edge_idx))

            state_p = torch.tensor(dp[3], dtype=torch.float32).to(self.device)  # 下一观测信息
            batch_list_p.append(Data(x=state_p, edge_index=self.edge_idx))
            rewards.append(dp[2])
            actions.append(dp[1])
        batch_t = Batch.from_data_list(batch_list)
        batch_tp = Batch.from_data_list(batch_list_p)
        # TODO reshape slow warning
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.device)
        if self.sub_agents > 1:
            rewards = rewards.view(rewards.shape[0] * rewards.shape[1])
            actions = actions.view(actions.shape[0] * actions.shape[1])  # TODO: check all dimensions here
        # rewards = rewards.view(rewards.shape[0] * rewards.shape[1])
        # actions = torch.tensor(np.array(actions), dtype=torch.long)
        # actions = actions.view(actions.shape[0] * actions.shape[1])  # TODO: check all dimensions here

        return batch_t, batch_tp, rewards, actions, act_log_prob

    def train(self, transition_buffer):
        # samples = random.sample(self.replay_buffer, self.batch_size)
        # b_t, b_tp, rewards, actions = self._batchwise(samples)
        max_episode_len = len(transition_buffer)
        b_t, b_tp, rewards, actions, act_log_prob = self._batchwise(transition_buffer)
        # act_log_prob = torch.tensor(np.array(act_log_prob), dtype=torch.float32).to(self.device)
        obs = b_t.x
        obs = obs.view(max_episode_len, self.sub_agents, -1)
        obs_next = b_tp.x
        out_next = self.model(x=b_tp.x, edge_index=b_tp.edge_index, train=False)
        target = rewards.view(-1, 1) + self.gamma * out_next.view(-1, 1)

        out = self.model(x=b_t.x, edge_index=b_t.edge_index, train=True)
        # td_error = target - out
        # advantage = compute_advantage(self.gamma, self.lmbda, td_error.cpu()).to(self.device)  # gae
        td_error = (target - out).view(max_episode_len, self.sub_agents, -1)
        td_error = td_error.permute(1, 0, 2)
        adv = torch.empty((0), dtype=torch.float32).to(self.device)
        for item in td_error:  # 每个交叉口的gae
            advantage = compute_advantage(self.gamma, self.lmbda, item.cpu()).to(self.device)  # gae
            adv = torch.cat((adv, advantage), 0)
        adv = adv.view(self.sub_agents, max_episode_len, 1)
        advantage = adv.permute(1, 0, 2)
        advantage = advantage.contiguous().view(-1, 1)
        actions = actions.view(max_episode_len, self.sub_agents, 1)
        actions = actions.type(torch.long)

        epochs = 15
        for _ in range(epochs):
            new_probs = torch.empty((0), dtype=torch.float32).to(self.device)
            rnn_state = None

            for item in obs:
                inp = item.to(self.device)
                new_act, rnn_state = self.actor(inp, rnn_state)
                new_probs = torch.cat((new_probs, new_act.probs), 0)
            # new_act, rnn_state = self.actor(obs, rnn_state)
            new_act_p = new_probs.view(max_episode_len, self.sub_agents, 8)
            new_log_probs = torch.log(torch.gather(new_act_p, dim=2, index=actions)).to(self.device)
            ratio = torch.exp(new_log_probs.view(-1, 1) - act_log_prob.detach().view(-1, 1))
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage
            # 策略熵
            log_new_act = torch.log(new_act_p)
            policy_entropy = - torch.sum(new_act_p * log_new_act, dim=2)
            actor_loss = torch.mean(-torch.min(surr1, surr2))# + 0.01 * torch.mean(policy_entropy)
            new_out = self.model(x=b_t.x, edge_index=b_tp.edge_index, train=True)
            critic_loss = torch.mean(F.mse_loss(new_out, target.detach()))

            self.critic_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            critic_loss.backward()
            actor_loss.backward()
            self.critic_optimizer.step()
            self.actor_optimizer.step()

            # # for name, parms in self.model.named_parameters():
            # #     print('-->name:', name)
            # #     print('-->para:', parms)
            # #     print('-->grad_requires:', parms.requires_grad)
            # #     print('-->grad_value:', parms.grad)
            # #     print("===")
            # if self.epsilon > self.epsilon_min:
            #     self.epsilon *= self.epsilon_decay
        torch.cuda.empty_cache()  # 释放显存
        return critic_loss.cpu().clone().detach().numpy()

    def update_target_network(self):
        weights = self.model.state_dict()
        self.target_model.load_state_dict(weights)

    def load_model(self, e):
        model_name = os.path.join(Registry.mapping['logger_mapping']['output_path'].path,
                                  'model', f'{e}_{self.rank}.pt')
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(model_name))
        self.target_model = self._build_model()
        self.target_model.load_state_dict(torch.load(model_name))

    def save_model(self, e):
        path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = os.path.join(path, f'{e}_{self.rank}.pt')
        torch.save(self.target_model.state_dict(), model_name)


class ColightNet(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(ColightNet, self).__init__()
        self.model_dict = kwargs
        self.action_space = gym.spaces.Discrete(output_dim)  # 8
        self.features = input_dim  # 12
        self.module_list = nn.ModuleList()
        self.embedding_MLP = Embedding_MLP(self.features, layers=self.model_dict.get('NODE_EMB_DIM'))
        for i in range(self.model_dict.get('N_LAYERS')):  # 0
            block = MultiHeadAttModel(d=self.model_dict.get('INPUT_DIM')[i],
                                      dv=self.model_dict.get('NODE_LAYER_DIMS_EACH_HEAD')[i],
                                      d_out=self.model_dict.get('OUTPUT_DIM')[i],
                                      nv=self.model_dict.get('NUM_HEADS')[i],
                                      suffix=i)
            self.module_list.append(block)
        output_dict = OrderedDict()

        if len(self.model_dict['OUTPUT_LAYERS']) != 0:
            # TODO: dubug this branch
            for l_idx, l_size in enumerate(self.model_dict['OUTPUT_LAYERS']):
                name = f'output_{l_idx}'
                if l_idx == 0:
                    h = nn.Linear(block.d_out, l_size)
                else:
                    h = nn.Linear(self.model_dict.get('OUTPUT_LAYERS')[l_idx - 1], l_size)
                output_dict.update({name: h})
                name = f'relu_{l_idx}'
                output_dict.update({name: nn.ReLU})
            out = nn.Linear(self.model_dict['OUTPUT_LAYERS'][-1], self.action_space.n)  # 128, 8
        else:
            out = nn.Linear(block.d_out, 1)  # in:128, out:8
        name = f'output'
        output_dict.update({name: out})  # 一个线性层128，8
        self.output_layer = nn.Linear(128, 1)
        self.dense_1 = nn.Linear(32, 32)

    def forward(self, x, edge_index, train=True):
        h = self.embedding_MLP.forward(x, train)
        # TODO: implement att
        # for mdl in self.module_list:  # mdl: MultiHeadAttModel()
        #     h = mdl.forward(h, edge_index, train)
        # if train:
        #     h = self.output_layer(h)
        # else:
        #     with torch.no_grad():
        #         h = self.output_layer(h)
        # h =F.relu(self.dense_1(h))
        h = self.output_layer(h)
        return h

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Embedding_MLP(nn.Module):
    def __init__(self, in_size, layers):  # layers: [128, 128]
        super(Embedding_MLP, self).__init__()
        # constructor_dict = OrderedDict()
        # for l_idx, l_size in enumerate(layers):  # 将列表组成索引0：128，0：128
        #     name = f"node_embedding_{l_idx}"
        #     if l_idx == 0:
        #         h = nn.Linear(in_size, l_size)
        #         constructor_dict.update({name: h})  # [('node_embedding_0', Linear(in_features=12, out_features=128, bias=True))]
        #     else:
        #         h = nn.Linear(layers[l_idx - 1], l_size)  # 128, 128
        #         constructor_dict.update({name: h})
        #     name = f"n_relu_{l_idx}"
        #     constructor_dict.update({name: nn.ReLU()})
            # constructor_dict: OrderedDict([('node_embedding_0', Linear(in_features=12, out_features=128, bias=True)), ('n_relu_0', ReLU()),
            # ('node_embedding_1', Linear(in_features=128, out_features=128, bias=True)), ('n_relu_1', ReLU())])
        hidden_dim = 128
        self.embedding_node = nn.Sequential(
            layer_init(nn.Linear(12, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU()
        )

    def _forward(self, x):
        x = self.embedding_node(x)
        return x

    def forward(self, x, train=True):
        if train:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)


class MultiHeadAttModel(MessagePassing):
    """
    inputs:
        In_agent [bacth,agents,128]
        In_neighbor [agents, neighbor_num]
        l: number of neighborhoods (in my code, l=num_neighbor+1,because l include itself)
        d: dimension of agents's embedding
        dv: dimension of each head
        dout: dimension of output
        nv: number of head (multi-head attention)
    output:
        -hidden state: [batch,agents,32]
        -attention: [batch,agents,neighbor]
    """
    def __init__(self, d, dv, d_out, nv, suffix):
        super(MultiHeadAttModel, self).__init__(aggr='add')
        self.d = d  # 128
        self.dv = dv  # 16
        self.d_out = d_out  # 128
        self.nv = nv  # 5
        self.suffix = suffix  # 0
        # target is center
        self.W_target = nn.Linear(d, dv * nv)  # 128, 80
        self.W_source = nn.Linear(d, dv * nv)
        self.hidden_embedding = nn.Linear(d, dv * nv)
        self.out = nn.Linear(dv, d_out)  # 16, 128
        self.att_list = []
        self.att = None

    def _forward(self, x, edge_index):
        # TODO: test batch is shared or not

        # x has shape [N, d], edge_index has shape [E, 2]
        edge_index, _ = add_self_loops(edge_index=edge_index)
        # 给出自己和邻居观测
        # x1 = torch.unsqueeze(x, dim=0)
        # x2 = x1.repeat(2, 1, 1)
        # edge1 = F.one_hot(edge_index)
        # # e = torch.mul(x, edge_index)
        # re = torch.bmm(edge1.float(), x2)

        aggregated = self.propagate(x=x, edge_index=edge_index)  # [16, 16] 到message执行
        # aggregated = self.message1(re[0], re[1], edge_index=edge_index)  # [16, 16] 到message执行
        out = self.out(aggregated)
        out = F.relu(out)  # [ 16, 128]
        # self.att = self.att_list
        return out

    def forward(self, x, edge_index, train=True):
        if train:
            return self._forward(x, edge_index)
        else:
            with torch.no_grad():
                return self._forward(x, edge_index)

    def message(self, x_i, x_j, edge_index):  # xi是路口的信息，xj是xi路口的邻居
        h_target = F.relu(self.W_target(x_i))  # 7, 80
        h_target = h_target.view(h_target.shape[:-1][0], self.nv, self.dv)  # 7, 5, 16  5是注意力头  16是每个头的维度，分别提取五个特征
        agent_repr = h_target.permute(1, 0, 2)  # 维度换位  5, 7, 16

        h_source = F.relu(self.W_source(x_j))  # 7, 80
        h_source = h_source.view(h_source.shape[:-1][0], self.nv, self.dv)  # 7, 5, 1
        neighbor_repr = h_source.permute(1, 0, 2)  # [nv, E, dv] 5, 7, 16

        index = edge_index[1]  # which is target  [(1,2,3),(2,1,3)]   tensor([1, 2, 0, 1, 0, 1, 2])

        e_i = torch.mul(agent_repr, neighbor_repr).sum(-1)  # [5, 64] 5, 7
        max_node = torch_scatter.scatter_max(e_i, index=index)[0]  # [5, 16] 5, 3  把index相同的ecexp_i相加 每一维有三种0，1，2
        max_i = max_node.index_select(1, index=index)  # [5, 64] 5, 7
        ec_i = torch.add(e_i, -max_i)  # 5, 7
        ecexp_i = torch.exp(ec_i)
        norm_node = torch_scatter.scatter_sum(ecexp_i, index=index)  # [5, 16] 5, 3 # index相同的ecexp_i相加
        normst_node = torch.add(norm_node, 1e-12)  # [5, 16] 5, 3
        normst_i = normst_node.index_select(1, index)  # [5, 64] 5, 7

        alpha_i = ecexp_i / normst_i  # [5, 64]  归一化的注意力分数
        alpha_i_expand = alpha_i.repeat(self.dv, 1, 1)  # 16, 5, 7
        alpha_i_expand = alpha_i_expand.permute((1, 2, 0))  # [5, 64, 16]  5, 7, 16

        hidden_neighbor = F.relu(self.hidden_embedding(x_j))  # 7, 80
        hidden_neighbor = hidden_neighbor.view(hidden_neighbor.shape[:-1][0], self.nv, self.dv)  # 7, 5, 16
        hidden_neighbor_repr = hidden_neighbor.permute(1, 0, 2)  # [5, 64, 16]  # 5, 7, 16
        out = torch.mul(hidden_neighbor_repr, alpha_i_expand).mean(0)  # 7, 16
        out_ = torch_scatter.scatter_sum(out, index=index, dim=0)

        # TODO: attention ouput in the future
        #self.att_list.append(alpha_i)  # [64, 16]
        return out

    def get_att(self):
        if self.att is None:
            print('invalid att')
        return self.att


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):  # 输入是原始观测  输出策略的概率
        super(Actor, self).__init__()
        self.embedding_MLP = Embedding_MLP(input_dim, layers=1)
        self.rnn_hidden = 128
        self.rnn = nn.GRUCell(self.rnn_hidden, self.rnn_hidden)
        # self.dense_1 = nn.Linear(input_dim, 64)
        # self.dense_2 = nn.Linear(64, 32)
        # self.dense_3 = nn.Linear(64, output_dim)

        self.dense_1 = nn.Linear(self.rnn_hidden, output_dim) # 加上 std=0.01
        # self.dense_1 = layer_init(nn.Linear(input_dim, 64))
        # self.dense_2 = layer_init(nn.Linear(64,  output_dim), std=0.01)

    def forward(self, x, rnn_state):
        # x = x.view(-1,12)
        x = self.embedding_MLP(x)
        if rnn_state is not None:
            rnn_state = rnn_state.view(-1, self.rnn_hidden)
        h = self.rnn(x, rnn_state)
        # x = F.relu( self.dense_1(x))  # 去掉了 , inplace=False
        x = self.dense_1(h)
        # x = F.relu(self.dense_3(x))
        # x = torch.tanh(self.dense_1(x))
        # x = torch.tanh(x)
        x = torch.distributions.Categorical(logits=x)
        return x, h

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)