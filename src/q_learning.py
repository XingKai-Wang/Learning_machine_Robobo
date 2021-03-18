import numpy as np
from math import inf 
from object_detection import selectHeading, getContours
from actions import *
from random import uniform, randint
import pandas as pd 
import robobo
import prey


class QLearning:
    def __init__(self, **args):
        self.epsilon = args.get('epsilon', 0.15)
        self.alpha = args.get('alpha',0.3)
        self.gamma = args.get('gamma', 0.9)
        self.n_actions = args.get('n_actions', 6)
        

    def calculateQValue(self, max_q, old_q, reward):
        return ((1-self.alpha)*(old_q) + self.alpha*(reward + self.gamma*(max_q)-old_q))


    def generateMoveFromPolicy(self, candidate_values):
        # print(candidate_values)
        if self.generateEpsilon():
            selected_move = self.generateRandomMove()
        else:
            selected_move = self.generateGreedyMove(candidate_values)
            
        return selected_move

    def generateEpsilon(self):
        if uniform(0,1) <= self.epsilon:
            return True
        return False
    
    def generateRandomMove(self):
        return randint(0,self.n_actions-1)

    def generateGreedyMove(self, candidate_values):
        return np.argmax(candidate_values)


class Agent:
    def __init__(self, rob, **args):
        # self.stuck_threshold = args.get('stuck_threshold',0.1)
        # self.q_table = args.get('q_table',np.zeros(shape=(2,2,2,2,2,2,2,2,2,6)))


        self.debug_print = args.get('debug_print', False)
        self.rob = rob
        # self.state_variables = ['robot_position', 'sensor_state','stuck_state', 'sensor_data' ]
        self.state_history = []
        self.data_history = []
        self.current_state =[]
        self.current_data = {}
        self.food_consumed = 0
        self.current_reward = 0
        self.reward_history = []
        self.q_table = args.get('q_table',np.random.rand(4,2,2,4,4,4,4,4,6))
        # self.full_stuck_count = 0
        self.current_action = None
        self.action_history = []
        # self.stuck_count = 0
        self.distance_threshold = args.get('distance_threshold', {'out_of_range': inf,
                                                                'far': 0.1,
                                                                'middle': 0.06,
                                                                'near': 0.001
                                                                   }
                                            )






   



    def playAction(self, action):
        selectMove(self.rob, action = action)
        self.current_action = action
        self.action_history.append(action)



    # def isFullStuck(self):
    #     flag = False
    #     if len(self.previous_data['robot_position']) <= 1:
    #         flag =  False

    #     else:
    #         curr = self.previous_data['robot_position'][-1]
    #         prev  = self.previous_data['robot_position'][-2]
    #         arr = np.absolute(np.subtract(curr ,prev))
    #         flag =  all(i <= 0.03 for i in arr)
            
    #     stuck = 0
    #     if any(self.current_data['sensor_state']) and flag:
    #         stuck = 1
    #     return stuck



    # def isStuck(self):
    #     if len(self.previous_data['robot_position']) <= 1:
    #         return 0
    #     curr = self.previous_data['robot_position'][-1]
    #     prev  = self.previous_data['robot_position'][-2]
    #     arr = np.absolute(np.subtract(curr ,prev))
    #     if self.debug_print:
    #         print(arr)
    #     return (int(all(i <= self.stuck_threshold for i in arr)))



    # def createSensorState(self, sensor_input):
    #     sensor_input = [1 if i>0 else 0 for i in sensor_input]
    #     return sensor_input

    def initAgent(self):
        self.rob.stop_world()
     
        

        self.state_history = []
        self.data_history = []
        self.current_state =[]
        self.current_data = {}

        self.current_reward = 0
        self.reward_history = []
        # self.full_stuck_count = 0
        self.current_action = None
        self.action_history = []
        # self.stuck_count = 0
        self.food_consumed = 0

        self.rob.play_simulation()
        self.rob.play_simulation()
        self.updateState()
        

        print('Agent Reset')

    def generateStateData(self):
        sensor_data = self.rob.read_irs()[3:]
        img = self.rob.get_image_front()

        # is_far = self.isFar(sensor_data)
        # is_food = self.isFood(img)


        sensor_data = [inf if i == False else i for i in sensor_data]

        sensor_state = self.createSensorState(sensor_data)

        food_heading, distance = selectHeading(img)
        # print('@@', food_heading)
        if food_heading == 0:
            is_food = 0
        else:
            is_food = 1
        # full_stuck = self.isFullStuck()
        # stuck = self.isStuck()
        # sensor_state = self.createSensorState(sensor_data)
        is_closer = self.isCloser(distance)
        # food_count = self.rob.collected_food()
        food_count = 0
        has_eaten = self.hasEaten(food_count)
        return {'sensor_data': sensor_data,
                'is_food': is_food,
                'food_heading': food_heading,
                # 'full_stuck': full_stuck,
                'sensor_state':sensor_state,
                'distance':distance,
                'is_closer':is_closer,
                'has_eaten':has_eaten,
                'food_count':food_count
                }

        
    def isCloser(self,distance):
        if len(self.data_history) < 1:
            return 0
        
        curr = distance
        prev = self.data_history[-1]['distance']
        if curr > prev:
            return 1
        else:
            return 0


    def isFood(self,img):
        result = getContours(img)
        if result:
            return 1
        return 0 

    def isFar(self, sensor_data):
        # _sensor_data = [inf if i == False else i for i in sensor_data]
        return not(any(sensor_data))



    def createSensorState(self, sensor_data):
        converted = []
        for i in sensor_data:
            if i == self.distance_threshold['out_of_range']:
                converted.append(3)
            if i < self.distance_threshold['out_of_range'] and i >= self.distance_threshold['far']:
                converted.append(2)
            elif i < self.distance_threshold['far'] and i >= self.distance_threshold['middle']:
                converted.append(1)
            elif i < self.distance_threshold['middle']:
                converted.append(0)

        return converted


    def updateState(self):
        state = self.generateStateData()


        # is_food = state.get('is_food', None)
        # full_stuck = state.get('full_stuck', None)
        
        self.current_data = state
        self.data_history.append(self.current_data)
        # is_closer = self.isCloser()


        self.current_state =  [self.current_data['food_heading']]+[self.current_data['is_closer']]+[self.current_data['has_eaten']]+self.current_data['sensor_state']


        
        self.state_history.append(self.current_state)
        # self.generateRewards()
        # self.stuck_count+=stuck
        # self.full_stuck_count+=full_stuck

    def executeMove(self,move):
        self.playAction(move)
        self.updateState()
        self.generateRewards()

    def hasEaten(self, food_count):
        if len(self.data_history) < 1:
            return 0
        
        prev = self.data_history[-1]['food_count']
        curr = food_count
        if curr > prev:
            return 1
        else:
            return 0

    def generateRewards(self):
        reward = 0
        is_food = self.current_data['is_food']
        if is_food:
            #reward for seeing food
            reward+=2
            # reward for seeing food and detecting something
            for i in self.current_data['sensor_state']:
                _reward = 3/(i+1)
                reward += _reward
        elif not is_food:
            #reward for not seeing food
            reward+=-2
            #reward for not seeing food and detecting something
            for i in self.current_data['sensor_state']:
                _reward = -3/(i+1)
                reward += _reward

        is_closer=self.current_data['is_closer']
        if is_closer:
            #reward for moving closer to food
            reward+=3

        has_eaten = self.current_data['has_eaten']
        if has_eaten:
            #reward for eating food
            reward+=5
        # reward = self.current_state[0]*-2

        # reward += sum(self.current_state[1:])*-1
    
        self.current_reward = reward
        
        self.reward_history.append(reward)

    def accessQTable(self, state):
        state = tuple(state)
        return self.q_table[state]

    def isCollision(self):
        sensor_state = self.current_data['sensor_state']

        #if there is anything right in front of sensor, sensor_flag is true
        sensor_flag = not(all(sensor_state))
        is_food = self.current_data['is_food']

        if sensor_flag and is_food:
            return 1

        return 0


def run(agent,q,n_episodes,max_steps,
        filename,max_food=7, load=False, save=True):
        if load:
            with open(filename,'rb') as f:
                agent.q_table = np.load(f)
        data = []
        for n in range(n_episodes):
            print(f'Episode {n}')
            agent.initAgent()
            print(agent.current_state)
            for i in range(max_steps):
                
                old_state = agent.current_state
                # print('old_state', old_state)
                next_move = q.generateMoveFromPolicy(agent.accessQTable(old_state))
                # print('q table', agent.accessQTable(old_state))
                # print('move', next_move,)
                agent.executeMove(next_move)
                new_state = agent.current_state
                old_q = agent.accessQTable(old_state)[next_move]
                max_q = np.max(agent.accessQTable(new_state))
                reward = agent.current_reward
                q_value = q.calculateQValue(max_q, old_q, reward)
                agent.q_table[tuple(old_state)][next_move] = q_value
                # food_collected = agent.rob.collected_food()
                food_collected = 0
                print(f'reward: {reward}, new_state: {new_state}, next_move: {next_move}, food_collect: {food_collected}')
                if food_collected == max_food:
                    print('All food collected')
                    break
                print(r'**********************************')
            _dict={}
            # _dict['food_collected'] = agent.rob.collected_food() 
            _dict['food_collected'] = 0
            _dict['total_reward'] = sum(agent.reward_history)
            _dict['steps'] = i
            data.append(_dict)
            print(_dict)
        print('Training Finished')
        if save:
            with open(filename, 'wb') as f:
                
                np.save(f, agent.q_table)
                print('Q table saved: ', filename)
            df = pd.DataFrame(data)
            filename = filename.split('.')[0]+'_train'+'.csv'
            df.to_csv(filename, index=False)
            print('data saved at ', filename)



def runPredatorPrey(agent,q,n_episodes,max_steps,
        filename,max_food=7, load=False, save=True):

        if load:
            with open(filename,'rb') as f:
                agent.q_table = np.load(f)
        data = []
        for n in range(n_episodes):
            print(f'Episode {n}')
            agent.initAgent()
            print('Starting prey')

            prey_controller, prey_robot = startPrey()
            print(agent.current_state)
            for i in range(max_steps):
                
                old_state = agent.current_state
                # print('old_state', old_state)
                next_move = q.generateMoveFromPolicy(agent.accessQTable(old_state))
                # print('q table', agent.accessQTable(old_state))
                # print('move', next_move,)
                agent.executeMove(next_move)
                new_state = agent.current_state
                old_q = agent.accessQTable(old_state)[next_move]
                max_q = np.max(agent.accessQTable(new_state))
                reward = agent.current_reward
                q_value = q.calculateQValue(max_q, old_q, reward)
                agent.q_table[tuple(old_state)][next_move] = q_value
                # food_collected = agent.rob.collected_food()
                food_collected = 0
                print(f'reward: {reward}, new_state: {new_state}, next_move: {next_move}, Prey caught: {agent.isCollision()}, Is food: {agent.current_data["is_food"]}, {not(all(agent.current_data["sensor_state"]))}')
                if agent.isCollision():
                    print('Prey caught')
                    break
                print(r'**********************************')
            stopPrey(prey_controller,prey_robot)
            _dict={}
            # _dict['food_collected'] = agent.rob.collected_food() 
            _dict['food_collected'] = 0
            _dict['total_reward'] = sum(agent.reward_history)
            _dict['steps'] = i
            data.append(_dict)
            print(_dict)
        print('Training Finished')
        if save:
            with open(filename, 'wb') as f:
                
                np.save(f, agent.q_table)
                print('Q table saved: ', filename)
            df = pd.DataFrame(data)
            filename = filename.split('.')[0]+'_train'+'.csv'
            df.to_csv(filename, index=False)
            print('data saved at ', filename)

            
def startPrey():

    prey_robot = robobo.SimulationRoboboPrey().connect(address='127.0.0.1', port=19989)
    
    prey_controller = prey.Prey(robot=prey_robot, level=0)

    prey_controller.start()
    return(prey_controller, prey_robot)

def stopPrey(prey_controller, prey_robot):
    prey_controller.stop()
    prey_controller.join()
    prey_robot.disconnect()