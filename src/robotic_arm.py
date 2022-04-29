from nuro_arm.robot.robot_arm import RobotArm

from connect_four_env import Connect4Env

from agent import RandomAgent
import numpy as np
import time
import sys

JOINT_POSITIONS = [
    [],  # 0
    [],  # 1
    [],  # 2
    [],  # 3
    [],  # 4
    [],  # 5
    [],  # 6
]

INTERMEDIATE_STATE = []


def play_game():
    robot = RobotArm()
    game = Connect4Env()
    agent = RandomAgent()
    robot_first = np.random.rand() < 0.5
    if robot_first:
        print('Robot is going first')
        action = agent(game)
        game.step(action)
        take_action(robot, action)


def take_action(robot, action):
    robot.move_arm_jpos(INTERMEDIATE_STATE)
    print('robot is trying to move to column ')
    ready = input('type yes if robot is ready to pick up piece, type anything else if not')
    if ready.lower() == 'yes':
        robot.set_gripper_state(0)
        time.sleep(1)
        robot.set_gripper_state(1)
    else:
        return False

    robot.move_arm_jpos(JOINT_POSITIONS[action])
    robot.set_gripper_state(1)

    success = input('type yes if robot was successful, type anything else if not')
    if success.lower() == 'yes':
        return True
    else:
        try_again = input('type yes if you want the robot to try again, type anything else if not')
        if try_again == 'yes':
            return take_action(robot, action)
        else:
            human_fix = input('type yes if you want to manually place the robot\'s piece, type anything else if not')
            if human_fix.lower() == 'yes':
                return True
            else:
                return False

if __name__ == '__main__':
    print(nuro_arm)
    print('passed')
