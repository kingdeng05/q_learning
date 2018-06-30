#!/usr/bin/env python

import numpy as np
import os

class QLearning:
    def __init__(self):
        self.n_iter = 1000 # number of episodes(iterations)
        self.gamma = 0.8
        self.env_mat = np.array([[-1, -1, -1, -1, 0, -1], \
                               [-1, -1, -1, 0, -1, 100], \
                               [-1, -1, -1, 0, -1, -1],
                               [-1, 0, 0, -1, 0, -1],
                               [0, -1, -1, 0, -1, 100],
                               [-1, 0, -1, -1, 0, 100]])
        self.q_mat = np.zeros((6, 6))
        self.learn()

    def learn(self):
        print 'learning...'
        for episode in range(self.n_iter):
            cur_s = self.select_state() 
            print 'in episode {}, init state {}'.format(episode, cur_s)
            while True:
                cur_a = self.select_action(cur_s)
                #print 'int state is {}, selected action is {}'.format(cur_s, cur_a)
                res = self.get_reward(cur_s, cur_a)
                next_s = cur_a
                move_res = self.get_best_move(next_s) 
                self.q_mat_upt(res + self.gamma * move_res, cur_s, cur_a)
                cur_s = next_s 
                if self.reach_goal(cur_s):
                    break
        self.q_mat_extract()

    def reach_goal(self, cur_s):
        return cur_s == 5

    def select_state(self):
        return np.random.choice(6, 1)[0]

    def select_action(self, s):
        a_list = self.env_mat[s]
        return np.random.choice(np.where(a_list != -1)[0], 1)[0] 

    def get_reward(self, s, a):
        return self.env_mat[s][a]

    def get_best_move(self,cur_s):
        return np.max(self.q_mat[cur_s])

    def q_mat_upt(self, res, s, a):
        self.q_mat[s][a] = res
        
    def q_mat_extract(self):
        print 'Training result'
        print '*************'
        for s in range(self.q_mat.shape[0]):
            print 'at state {}, best move is {}'.format(s, np.argmax(self.q_mat[s]))
        print '*************'
        print 'Q matrix is:'
        print self.q_mat

if __name__ == "__main__":
   _ = QLearning() 
