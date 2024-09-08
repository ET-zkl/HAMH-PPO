import os
import numpy as np
from common.metrics import Metrics
from environment import TSCEnv
from common.registry import Registry
from trainer.base_trainer import BaseTrainer
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA

@Registry.register_trainer("tsc")
class TSCTrainer(BaseTrainer):
    '''
    Register TSCTrainer for traffic signal control tasks.
    '''
    def __init__(
        self,
        logger,
        gpu=0,
        cpu=False,
        name="tsc"
    ):
        super().__init__(
            logger=logger,
            gpu=gpu,
            cpu=cpu,
            name=name
        )
        self.episodes = Registry.mapping['trainer_mapping']['setting'].param['episodes']
        self.steps = Registry.mapping['trainer_mapping']['setting'].param['steps']
        self.test_steps = Registry.mapping['trainer_mapping']['setting'].param['test_steps']
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.action_interval = Registry.mapping['trainer_mapping']['setting'].param['action_interval']  # 10
        self.save_rate = Registry.mapping['logger_mapping']['setting'].param['save_rate']
        self.learning_start = Registry.mapping['trainer_mapping']['setting'].param['learning_start']
        self.update_model_rate = Registry.mapping['trainer_mapping']['setting'].param['update_model_rate']
        self.update_target_rate = Registry.mapping['trainer_mapping']['setting'].param['update_target_rate']
        self.test_when_train = Registry.mapping['trainer_mapping']['setting'].param['test_when_train']
        # replay file is only valid in cityflow now. 
        # TODO: support SUMO and Openengine later
        
        # TODO: support other dataset in the future
        self.dataset = Registry.mapping['dataset_mapping'][Registry.mapping['command_mapping']['setting'].param['dataset']](
            os.path.join(Registry.mapping['logger_mapping']['path'].path,
                         Registry.mapping['logger_mapping']['setting'].param['data_dir'])
        )
        self.dataset.initiate(ep=self.episodes, step=self.steps, interval=self.action_interval)
        self.yellow_time = Registry.mapping['trainer_mapping']['setting'].param['yellow_length']  # 5
        # consists of path of output dir + log_dir + file handlers name
        self.log_file = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                     Registry.mapping['logger_mapping']['setting'].param['log_dir'],
                                     os.path.basename(self.logger.handlers[-1].baseFilename).rstrip('_BRF.log') + '_DTL.log'
                                     )

    def create_world(self):
        '''
        create_world
        Create world, currently support CityFlow World, SUMO World and Citypb World.

        :param: None
        :return: None
        '''
        # traffic setting is in the world mapping
        self.world = Registry.mapping['world_mapping'][Registry.mapping['command_mapping']['setting'].param['world']](
            self.path, Registry.mapping['command_mapping']['setting'].param['thread_num'],interface=Registry.mapping['command_mapping']['setting'].param['interface'])

    def create_metrics(self):
        '''
        create_metrics
        Create metrics to evaluate model performance, currently support reward, queue length, delay(approximate or real) and throughput.

        :param: None
        :return: None
        '''
        if Registry.mapping['command_mapping']['setting'].param['delay_type'] == 'apx':
            lane_metrics = ['rewards', 'queue', 'delay']
            world_metrics = ['real avg travel time', 'throughput']
        else:
            lane_metrics = ['rewards', 'queue']
            world_metrics = ['delay', 'real avg travel time', 'throughput']
        self.metric = Metrics(lane_metrics, world_metrics, self.world, self.agents)

    def create_agents(self):
        '''
        create_agents
        Create agents for traffic signal control tasks.

        :param: None
        :return: None
        '''
        self.agents = []
        print(Registry)
        agent = Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](self.world, 0)
        print(agent)
        num_agent = int(len(self.world.intersections) / agent.sub_agents)
        # num_agent = len(self.world.intersections)
        self.agents.append(agent)  # initialized N agents for traffic light control
        for i in range(1, num_agent):
            self.agents.append(Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](self.world, i))

        # for magd agents should share information 
        if Registry.mapping['model_mapping']['setting'].param['name'] == 'magd':
            for ag in self.agents:
                ag.link_agents(self.agents)

    def create_env(self):
        '''
        create_env
        Create simulation environment for communication with agents.

        :param: None
        :return: None
        '''
        # TODO: finalized list or non list
        self.env = TSCEnv(self.world, self.agents, self.metric)

    def train(self):
        '''
        train
        Train the agent(s).

        :param: None
        :return: None
        '''
        start = time.time()
        total_decision_num = 0
        flush = 0
        ep = 0
        r = []
        travel = []
        loss = []
        t = []
        for e in range(self.episodes):
            nnn = 0
            mmm = 0
            hyper_prob_1 = []
            hyper_prob_11 = []
            buffer_cur = []
            obs_arr = []
            # TODO: check this reset agent
            self.metric.clear()  # 评价指标
            #last_obs：[array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)]
            last_obs = self.env.reset()  # agent * [sub_agent, feature]  （3，12）

            for a in self.agents:
                a.reset()
            if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
                if self.save_replay and e % self.save_rate == 0:
                    self.env.eng.set_save_replay(True)
                    self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"episode_{e}.txt"))
                else:
                    self.env.eng.set_save_replay(False)
            i = 0
            rnn_state = None
            while i < self.steps:
                if i % self.action_interval == 0:  # action_interval = 10

                    last_phase = np.stack([ag.get_phase() for ag in self.agents])  # [agent, intersections]
                    actions = []
                    for idx, ag in enumerate(self.agents):
                        action, act_log_probs, rnn_state, w = ag.get_action(last_obs[idx],last_phase[idx], rnn_state, test=False)
                        actions.append(action)
                    actions = np.stack(actions)  # [agent, intersections]

                    rewards_list = []
                    for _ in range(self.action_interval):  # 时间步：10
                        obs, rewards, dones, _ = self.env.step(actions.flatten())
                        i += 1
                        rewards_list.append(np.stack(rewards))
                    rewards = np.mean(rewards_list, axis=0)  #求10个时间步的平均回报 [agent, intersection]
                    self.metric.update(rewards)  # 更新回报、队列、延迟指标

                    cur_phase = np.stack([ag.get_phase() for ag in self.agents])
                    for idx, ag in enumerate(self.agents):
                        buffer_cur.append((f'{e}_{i//self.action_interval}_{ag.id}', (last_obs[idx], last_phase[idx], actions[idx], rewards[idx], obs[idx], act_log_probs, w, cur_phase[idx])))

                    total_decision_num += 1
                    last_obs = obs  # 下一状态的观测)
                if all(dones):
                    break
            cur_loss_q = np.stack([ag.train(buffer_cur) for ag in self.agents])  # TODO: training
            loss.append(cur_loss_q)  # 1000步之后才训练
            ep = ep+1
            r.append(self.metric.rewards())
            travel.append(self.metric.real_average_travel_time())
            # print('episode:'+e+'/'+self.episodes+',q_loss:'+string(cur_loss_q)+',rewards:'+int(self.metric.rewards())+', real_average_travel_time:'+int(self.metric.real_average_travel_time()))
            # self.writeLog("episode:{}/{}, q_loss:{}, rewards:{}, throughput:{}, real_average_travel_time:{}".format(e, self.episodes,\
            #     int(cur_loss_q), self.metric.rewards(), int(self.metric.throughput()), self.metric.real_average_travel_time()))
            self.logger.info("episode:{}/{}, real avg travel time:{}, rewards:{}".format(e, self.episodes, self.metric.real_average_travel_time(), self.metric.rewards()))
            t.append(time.time()-start)

        end = time.time()
        print('代码执行时间：', end - start)
        plt.figure(1)
        plt.plot(np.arange(ep), r)
        plt.xlabel('episode')
        plt.ylabel('rewards')

        plt.figure(2)
        plt.plot(np.arange(ep), loss, 'red')
        plt.xlabel('episode')
        plt.ylabel('loss')
        plt.title('多策略_loss')
        plt.show()
        np.savetxt('PPOHyper_reward.txt', r)
        np.savetxt('PPOHyper_traveltime.txt', travel)
        np.savetxt('time.txt', t)
        # [ag.save_model(e=self.episodes) for ag in self.agents]
        self.logger.info("Final Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d" % (self.metric.real_average_travel_time(), \
        self.metric.rewards(), self.metric.queue(), self.metric.delay(), self.metric.throughput()))


    def train_test(self, e):
        '''
        train_test
        Evaluate model performance after each episode training process.

        :param e: number of episode
        :return self.metric.real_average_travel_time: travel time of vehicles
        '''
        obs = self.env.reset()
        self.metric.clear()
        for a in self.agents:
            a.reset()
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                actions = []
                for idx, ag in enumerate(self.agents):
                    actions.append(ag.get_action(obs[idx], phases[idx], test=True))
                actions = np.stack(actions)
                rewards_list = []
                # print(actions)
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions.flatten())  # make sure action is [intersection]
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric.update(rewards)
                # print('######')
                # print(rewards)
            if all(dones):
                break
        self.logger.info("Test step:{}/{}, travel time :{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(\
            e, self.episodes, self.metric.real_average_travel_time(), self.metric.rewards(),\
            self.metric.queue(), self.metric.delay(), int(self.metric.throughput())))
        self.writeLog("TEST", e, self.metric.real_average_travel_time(),\
            100, self.metric.rewards(),self.metric.queue(),self.metric.delay(), self.metric.throughput())
        return self.metric.real_average_travel_time()

    def test(self, drop_load=True):
        '''
        test
        Test process. Evaluate model performance.

        :param drop_load: decide whether to load pretrained model's parameters
        :return self.metric: including queue length, throughput, delay and travel time
        '''
        if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
            if self.save_replay:
                self.env.eng.set_save_replay(True)
                self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"final.txt"))
            else:
                self.env.eng.set_save_replay(False)
        self.metric.clear()
        r_step = []
        if not drop_load:
            [ag.load_model(self.episodes) for ag in self.agents]
        attention_mat_list = []
        obs = self.env.reset()
        for a in self.agents:
            a.reset()
        rnn_state = None
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                actions = []
                for idx, ag in enumerate(self.agents):
                    action, act_log_probs, rnn_state, w = ag.get_action(obs[idx],phases[idx], rnn_state, test=False)
                    actions.append(action)
                actions = np.stack(actions)  # [agent, intersections]
                # for idx, ag in enumerate(self.agents):
                #     actions.append(ag.get_action(obs[idx], phases[idx], test=True))
                # actions = np.stack(actions)
                rewards_list = []
                for j in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric.update(rewards)
                r_step.append(self.metric.rewards())
            if all(dones):
                break
        self.logger.info("Final Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d" % (self.metric.real_average_travel_time(), \
            self.metric.rewards(), self.metric.queue(), self.metric.delay(), self.metric.throughput()))
        plt.figure(1)
        plt.plot(np.arange(len(r_step)), r_step, 'red')
        plt.xlabel('episode')
        plt.ylabel('delay')
        plt.title('PPO_delay')
        plt.show()
        return self.metric

    def writeLog(self, mode, step, travel_time, loss, cur_rwd, cur_queue, cur_delay, cur_throughput):
        '''
        writeLog
        Write log for record and debug.

        :param mode: "TRAIN" or "TEST"
        :param step: current step in simulation
        :param travel_time: current travel time
        :param loss: current loss
        :param cur_rwd: current reward
        :param cur_queue: current queue length
        :param cur_delay: current delay
        :param cur_throughput: current throughput
        :return: None
        '''
        res = Registry.mapping['model_mapping']['setting'].param['name'] + '\t' + mode + '\t' + str(
            step) + '\t' + "%.1f" % travel_time + '\t' + "%.1f" % loss + "\t" +\
            "%.2f" % cur_rwd + "\t" + "%.2f" % cur_queue + "\t" + "%.2f" % cur_delay + "\t" + "%d" % cur_throughput
        log_handle = open(self.log_file, "a")
        log_handle.write(res + "\n")
        log_handle.close()

