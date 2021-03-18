
from __future__ import print_function


import time
import numpy as np
import traceback

import robobo
import cv2
import sys
import signal
import prey
from states import Agent
from actions import*
from q_learning import *
def terminate_program(signal_number, frame):
        print("Ctrl-C received, terminating program")
        sys.exit()
        rob.stop_world()
        sys.exit(1)

def simulation(rob):
    ## Initialize agent

    rob.set_phone_tilt(np.pi/5.5, 100)
    agent = Agent(rob=rob)
    ql = QLearning(epsilon=0.075)
    ## Train agent with specified parameters and save the controller
    # agent.train(filename = 'controller.npy', n_episodes = 39, max_steps = 60, shuffle=False)
    ## Load controller and run
    runPredatorPrey(agent = agent,q= ql,n_episodes = 1, max_steps = 10,max_food = 6, 
                            filename = r'F:\Documents\VU\LM\learning_machines_robobo\food_3.npy', load=True, save=False)

    rob.stop_world()


def main():
    # signal.signal(signal.SIGINT, terminate_program)

    # rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.7")
    global rob
    rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)
    
    try:
        simulation(rob)
        # test(rob)
    except Exception as e:
        rob.pause_simulation()
        cv2.imshow('img',rob.get_image_front())
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(e)
        traceback.print_exc()
        rob.stop_world()

    

    # Following code moves the robot

   
    # print("robobo is at {}".format(rob.position()))
    # # rob.sleep(1)

    # # # Following code moves the phone stand
    # # rob.set_phone_pan(343, 100)
    # # rob.set_phone_tilt(109, 100)
    # # time.sleep(1)
    # # rob.set_phone_pan(11, 100)
    # # rob.set_phone_tilt(26, 100)

    # # Following code makes the robot talk and be emotional
    # # rob.set_emotion('happy')
    # # rob.talk('Hi, my name is Robobo')
    # # rob.sleep(1)
    # # rob.set_emotion('sad')

    # # Following code gets an image from the camera
    # # image = rob.get_image_front()
    # # cv2.imwrite("test_pictures.png",image)

    # # time.sleep(0.1)

    # # IR reading
    # for i in range(2):
    #     print(rob.read_irs())
    #     # print("ROB Irs: {}".format(np.log(np.array(rob.read_irs()))/10))
    #     time.sleep(0.1)

    # # pause the simulation and read the collected food
    # # rob.pause_simulation()
    
    # # Stopping the simualtion resets the environment
    # rob.stop_world()
# def startPrey():

#     prey_robot = robobo.SimulationRoboboPrey().connect(address='127.0.0.1', port=19989)

#     prey_controller = prey.Prey(robot=prey_robot, level=2)

#     prey_controller.start()
#     return(prey_controller, prey_robot)

# def stopPrey(prey_controller, prey_robot):
#     prey_controller.stop()
#     prey_controller.join()
#     prey_robot.disconnect()


def test():
    rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)

    rob.play_simulation()

    # prey_robot = robobo.SimulationRoboboPrey().connect(address='127.0.0.1', port=19989)

    # prey_controller = prey.Prey(robot=prey_robot, level=2)

    # prey_controller.start()

    prey_controller, prey_robot = startPrey()

    for i in range(10):
            print("robobo is at {}".format(rob.position()))
            rob.move(5, 5, 2000)
    
    stopPrey(prey_controller, prey_robot)
    rob.stop_world()




if __name__ == "__main__":
    main()
    # test()






