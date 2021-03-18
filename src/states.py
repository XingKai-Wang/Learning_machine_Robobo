from pandas.core.reshape.reshape import stack_multiple
import numpy as np
from random import randint, uniform
from actions import *
import pandas as pd
import robobo

class Agent:
    def __init__(self,rob,**args):
        
        self.rob = rob
        self.state_variables = ['robot_position', 'sensor_state','stuck_state', 'sensor_data' ]
        self.previous_states = []
        self.previous_data = {}
        self.current_state =[]
        self.current_data = {}
        self.stuck_threshold = args.get('stuck_threshold',0.1)
        self.debug_print = args.get('debug_print', False)
        self.current_reward = 0
        self.reward_history = []
        self.q_table = args.get('q_table',np.zeros(shape=(2,2,2,2,2,2,2,2,2,6)))
        self.alpha = args.get('alpha',0.3)
        self.gamma = args.get('gamme', 0.9)
        self.full_stuck = 0

        self.epsilon = args.get('epsilon', 0.4)

        self.current_action = None
        self.action_history = []
        self.stuck_count = 0
        self.learn = True

        for i in self.state_variables:
            self.previous_data[i] = []

    def createSensorState(self, sensor_input):
        sensor_input = [1 if i>0 else 0 for i in sensor_input]
        return sensor_input

    def generateMoveFromPolicy(self):
        if self.generateEpsilon():
            selected_move = self.generateRandomMove()
        else:
            selected_move = self.generateGreedyMove()
            
        return selected_move

    def generateEpsilon(self):
        if uniform(0,1) <= self.epsilon:
            return True
        return False
    
    def generateRandomMove(self):
        return randint(0,5)

    def generateGreedyMove(self):
        # state = np.array(self.current_state)
        candidate_values = self.q_table[tuple(self.current_state)]
        return np.argmax(candidate_values)

    def calculateQValue(self, max_q, old_q, reward):
        return ((1-self.alpha)*(old_q) + self.alpha*(reward + self.gamma*(max_q)-old_q))

    def updateState(self, **state):
        self.current_action = state.get('action',None)
        self.action_history.append(self.current_action)
        state['sensor_state'] = self.createSensorState(state.get('sensor_data',None))
        for i in self.state_variables:
            if i == 'stuck_state':
                _state = self.isStuck()
                self.stuck_count += _state
            else: 
                _state = state.get(i, None)
            self.current_data[i] = _state
            self.previous_data[i].append(_state)
            # print(self.current_data)
        self.current_state = [self.current_data['stuck_state']] + self.current_data['sensor_state']
        self.previous_states.append(self.current_data)
        self.full_stuck+=self.isFullStuck()
        self.generateRewards()
        # print(self.current_reward)
        # print(self.current_state)
        # self.updateQTable()

    def isStuck(self):
        if len(self.previous_data['robot_position']) <= 1:
            # print(000)
            # return False
            return 0
        # print(1,self.previous_states['robot_position'])
        curr = self.previous_data['robot_position'][-1]
        prev  = self.previous_data['robot_position'][-2]
        arr = np.absolute(np.subtract(curr ,prev))
        # print(arr)
        # print(2)
        if self.debug_print:
            print(arr)
        return (int(all(i <= self.stuck_threshold for i in arr)))

    def generateRewards(self):
        # reward for staying stuck -2
        # reward for having detected something in sensor -1
        # print(self.current_state)
        reward = self.current_state[0]*-2

        reward += sum(self.current_state[1:])*-1
    
        self.current_reward = reward
        
        self.reward_history.append(reward)

    # def updateQTable(self, old_q, max_q, reward):
        


    def runEpisode(self):
        
        next_move = self.generateMoveFromPolicy()
        self.executeMove(next_move)

    # def readState(self):
    #     sensor_data = self.rob.read_irs()
    #     current_data = {'robot_position':self.rob.position(),
    #                         # 'sensor_state':sensor_input,
    #                         'sensor_data':sensor_data,
    #                         'action':move
    #                         }
    #     self.updateState(**current_data)

    def executeMove(self, move):
        old_state = self.current_state
        selectMove(self.rob, action = move)
        # self.readState()
        sensor_data = self.rob.read_irs()
        current_data = {'robot_position':self.rob.position(),
                            # 'sensor_state':sensor_input,
                            'sensor_data':sensor_data,
                            'action':move
                            }
        self.updateState(**current_data)

        if self.learn == True:
            new_state = self.current_state
            reward = self.current_reward
            # print(old_state,move)
            old_q = self.q_table[tuple(old_state)][move]
            max_q = np.max(self.q_table[tuple(new_state)])
            
            self.q_table[tuple(old_state)][move] = self.calculateQValue(max_q=max_q, old_q=old_q, reward=reward)


    def initEnv(self):
        self.previous_states = []
        self.previous_data = {}
        self.current_state =[]
        self.current_data = {}
        self.current_action = None
        self.action_history = []
        self.current_reward = 0
        self.full_stuck = 0
        self.reward_history = []
        for i in self.state_variables:
            self.previous_data[i] = []
        self.rob.stop_world()

        self.rob.play_simulation()
        self.rob.play_simulation()
        print('Environment Reset')

            # try:
            #     self.rob.shuffle_obstacles()
            # except:
            #     pass
        self.current_state = [0,0,0,0,0,0,0,0,0]
        self.stuck_count = 0
    def isFullStuck(self):
        flag = False
        if len(self.previous_data['robot_position']) <= 1:
            # print(000)
            # return False
            flag =  False
        # print(1,self.previous_states['robot_position'])
        else:
            curr = self.previous_data['robot_position'][-1]
            prev  = self.previous_data['robot_position'][-2]
            arr = np.absolute(np.subtract(curr ,prev))
            flag =  all(i <= 0.03 for i in arr)
            
        stuck = 0
        if any(self.current_data['sensor_state']) and flag:
            stuck = 1
        return stuck

    def train(self, n_episodes, max_steps, filename = None, shuffle = False):
        data = []
        for _ in range(n_episodes):
            print(f'Episode {_}')
            robbo_num = None
            if shuffle == True:
                self.rob.disconnect()
                robbo_num = _%3
                if robbo_num != 1:
                    robbo_num = f'#{robbo_num}'
                else:
                    robbo_num = ''
                # print(robbo_num)
                self.rob = robobo.SimulationRobobo(robbo_num).connect(address='127.0.0.1', port=19997)
            self.initEnv()
            first_move = self.generateRandomMove()
            self.executeMove(first_move)
            # print(first_move,self.current_state, self.current_reward)
            
            for i in range(max_steps):

                next_move = self.generateMoveFromPolicy()
                self.executeMove(next_move)
                print(self.current_action,self.current_state, self.current_reward, self.stuck_count, self.full_stuck)

            df = pd.DataFrame(self.previous_data)
            df['sensor_max'] = df['sensor_data'].apply(lambda x:np.max(x))
            df['sensor_mean'] = df['sensor_data'].apply(lambda x: np.mean(x))
            _dict = {}
            _dict['total_reward'] = sum(self.reward_history)
            _dict['stuck_count'] = self.stuck_count
            _dict['sensor_mean'] = np.mean(df['sensor_mean'].values)
            _dict['sensor_max'] = np.mean(df['sensor_max'].values)
            _dict['robbo'] = robbo_num
            _dict['full_stuck_count'] = self.full_stuck
            print(_dict)
            data.append(_dict)
        self.rob.stop_world()
        
        if filename:
            with open(filename, 'wb') as f:
                print('Training Finished')
                np.save(f, self.q_table)
                print('Q table saved: ', filename)
            df = pd.DataFrame(data)
            filename = filename.split('.')[0]+'_train'+'.csv'
            df.to_csv(filename, index=False)
            print('data saved at ', filename)

        
    def run(self, iterations, filename):
        
        with open(filename,'rb') as f:
            self.q_table = np.load(f)
        self.train(n_episodes=1, max_steps=iterations, shuffle=False, filename=None)


