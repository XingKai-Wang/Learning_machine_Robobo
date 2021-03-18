#!/usr/bin/env python2
from __future__ import print_function

import time

import robobo
import cv2
import sys
import signal
import prey


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

def main():
    signal.signal(signal.SIGINT, terminate_program)

    rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)

    rob.play_simulation()

    prey_robot = robobo.SimulationRoboboPrey().connect(address='127.0.0.1', port=19989)

    prey_controller = prey.Prey(robot=prey_robot, level=2)

    prey_controller.start()


    for i in range(10):
            print("robobo is at {}".format(rob.position()))
            rob.move(5, 5, 2000)
    
    prey_controller.stop()
    prey_controller.join()
    prey_robot.disconnect()
    rob.stop_world()

    # time.sleep(10)
    # rob.kill_connections()
    # rob = robobo.SimulationRobobo().connect(address='192.168.1.6', port=19997)
    rob.play_simulation()
    # prey_robot = robobo.SimulationRoboboPrey().connect(address='192.168.1.71', port=19989)
    # prey_controller = prey.Prey(robot=prey_robot, level=2, hardware=True)
    prey_robot = robobo.SimulationRoboboPrey().connect(address='127.0.0.1', port=19989)
    prey_controller = prey.Prey(robot=prey_robot, level=2)

    prey_controller.start()
    for i in range(10):
            print("robobo is at {}".format(rob.position()))
            rob.move(5, 5, 2000)
    prey_controller.stop()
    prey_controller.join()

    rob.stop_world()



if __name__ == "__main__":
    main()
