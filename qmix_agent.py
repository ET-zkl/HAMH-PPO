import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixer(nn.Module):
    def __init__(self,input_dim):
        super(QMixer, self).__init__()

        # self.args = args
        self.n_action = input_dim
        # self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = 32

        # self.abs = getattr(self.args, 'abs', True)

        # if getattr(args, "hypernet_layers", 1) == 1:
        self.hyper_w_1 = nn.Linear(12, self.embed_dim)
        self.hyper_w_final = nn.Linear(input_dim, self.embed_dim)
        # elif getattr(args, "hypernet_layers", 1) == 2:
        #     hypernet_embed = self.args.hypernet_embed
        #     self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
        #                                    nn.ReLU(inplace=True),
        #                                    nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
        #     self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
        #                                    nn.ReLU(inplace=True),
        #                                    nn.Linear(hypernet_embed, self.embed_dim))

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(12, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(input_dim, self.embed_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.embed_dim, 1))
        

    def forward(self, agent_qs, value_leader):  # 64,3,1/192,12

        # agent_qs = agent_qs.reshape(192, 1, 1)  # 192,1,1
        # bs = agent_qs.size(0)#192
        # # First layer
        # w1 = self.hyper_w_1(value_leader).abs()
        # b1 = self.hyper_b_1(value_leader)
        # w1 = w1.view(-1, 1, self.embed_dim)  # 64,3,32
        # b1 = b1.view(-1, 1, self.embed_dim)
        # hidden = F.elu(th.bmm(agent_qs, w1) + b1)  #  64, 1, 32
        #
        # # Second layer
        # w_final = self.hyper_w_final(value_leader).abs()
        # w_final = w_final.view(-1, self.embed_dim, 1)  #64, 32,1
        # # State-dependent bias
        # v = self.V(value_leader).view(-1, 1, 1)
        # # Compute final output
        # y = th.bmm(hidden, w_final) + v
        # # Reshape and return
        # q_tot = y.view(bs, -1)

        agent_qs = agent_qs.reshape(64, 1, 3)  # 64,1,3
        bs = agent_qs.size(0)  # 64
        # First layer
        t = torch.tensor(np.zeros((64, 1, 32)))
        t1 = t.type(torch.long)
        w1 = self.hyper_w_1(value_leader).abs()
        b1 = self.hyper_b_1(value_leader)
        w1 = w1.view(-1, 3, self.embed_dim)  # 64,3,32
        b1 = b1.view(-1, 3, self.embed_dim).gather(1, t1)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)  # 64, 1, 32

        # Second layer
        t = torch.tensor(np.zeros((64, 32, 1)))
        t1 = t.type(torch.long)
        w_final = self.hyper_w_final(value_leader).abs()
        w_final = w_final.view(-1, self.embed_dim, 1).gather(1, t1)  # 64, 32,1
        # State-dependent bias
        t = torch.tensor(np.zeros((64, 1, 1)))
        t1 = t.type(torch.long)
        v = self.V(value_leader).view(-1, 1, 1).gather(1,t1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1)
        
        return q_tot  # 192,1

    def k(self, states):
        bs = states.size(0)
        w1 = th.abs(self.hyper_w_1(states))
        w_final = th.abs(self.hyper_w_final(states))
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        w_final = w_final.view(-1, self.embed_dim, 1)
        k = th.bmm(w1,w_final).view(bs, -1, self.n_agents)
        k = k / th.sum(k, dim=2, keepdim=True)
        return k

    def b(self, states):
        bs = states.size(0)
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        b1 = self.hyper_b_1(states)
        b1 = b1.view(-1, 1, self.embed_dim)
        v = self.V(states).view(-1, 1, 1)
        b = th.bmm(b1, w_final) + v
        return b
