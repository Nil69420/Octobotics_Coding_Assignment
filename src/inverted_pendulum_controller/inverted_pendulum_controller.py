#!/usr/bin/env python3

####WORK IN PROGRESS####

from inverted_pendulum_sim.msg import CurrentState
from inverted_pendulum_sim.msg import ControlForce
from inverted_pendulum_sim.srv import SetParams, SetParamsResponse, SetParamsRequest
import rospy
import control
import math
import numpy as np

# Got these constants from Inverted Pendulum Sim Node
m = 10
M = 10
g = 9.8
l = 200

# Random A and B for testing
A = np.matrix([ [0, 1,                                       0, 0],
                [0, 0,            (-12 * m * g)/((12 * M) + m), 0],
                [0, 0,                                       0, 1],
                [0, 0, (12 * g * (M + m))/(l * ((13 * M) + m)), 0]
                ])

B = np.matrix([ [0],
                [13 / ((13 * M) + m)],
                [0],
                [-12 / (l * ((13 * M) + m))]
                ])

Q = np.diag( [1,1,1,1.] ) * 75
R = np.diag( [1.] )

# Use control module to get optimal gains
K, S, E = control.lqr( A, B, Q, R )

class LQR_InvertedPendulum_Controller:
    def __init__(self):
        rospy.init_node('LQR_Controller')
        # Publishes the current state of the inverted pendulum at 100 Hz
        self.command_pub = rospy.Publisher("/inverted_pendulum/current_state",
                                            CurrentState, queue_size=10)
        # Subscribes to the control force input to the inverted pendulum
        self.theta_sub = rospy.Subscriber("/inverted_pendulum/control_force",
                                          ControlForce, self.theta_callback)
        # Sets the parameters and initial conditions of the inverted pendulum
        self.pos_srv = rospy.Service("/inverted_pendulum/set_params",
                                        SetParams, self.pos_callback)
        self.current_state = np.array([0., 0., 0., 0.])
        self.desired_state = np.array([0., 0., 0., 0.])
        self.command_msg = ControlForce
    
    def theta_callback(self, theta_msg):
        # Callback to update angular state variables
        self.current_state[2] = theta_msg.process_value
        self.current_state[3] = theta_msg.process_value_dot
        rospy.loginfo_throttle(2, f'Current Angle: {math.degrees(theta_msg.process_value)}')
        
    def pos_callback(self, pos_msg):
        # Callback to update linear state variables
        self.current_state[0] = pos_msg.position[1]
        self.current_state[1] = pos_msg.velocity[1]
        
    def balance(self):
        # Get control output by multiplying K with state error
        self.command_msg.data = np.matmul(K, (self.desired_state - self.current_state))
        self.command_pub.publish(self.command_msg)
        rospy.loginfo_throttle(2, f'Commanding: {self.command_msg.data}')

def main():
    b = LQR_InvertedPendulum_Controller()
    while not rospy.is_shutdown():
        b.balance()          

if __name__ == '__main__':
    main()