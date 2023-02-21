# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 13:25:59 2018

@author: Lea
"""

import numpy as np


def ViterbiCalc(initial_probabilities,emission_probabilities,observations,transition_probability,states = None):
    if not isinstance(observations,np.ndarray):
        observations = np.array(observations)
        
    num_states = initial_probabilities.shape[0]
    num_observations = observations.shape[0]

#initial probabilities - 1    

    if states == None:
        states = range(num_states)
        
    V = np.zeros((num_states,num_observations))
    T = np.zeros((num_states,num_observations),dtype = int)
    for curr_state_index,state in enumerate(states):            
        V[curr_state_index,0] = initial_probabilities[curr_state_index]*emission_probabilities[curr_state_index,observations[0]]
        T[curr_state_index,0] = 0
 
#prbabilities - 2
    for i,observation in enumerate(observations[1:]):
         i += 1
         for curr_state_index,state in enumerate(states):
             max_prob = -1
             for prev_state_index in range(num_states):
                 prob = V[prev_state_index,i-1]*transition_probability[prev_state_index,curr_state_index]*emission_probabilities[curr_state_index,observation]
                 if prob > max_prob:
                     max_prob = prob
                     best_prev_state = prev_state_index
             V[curr_state_index,i] = max_prob
             T[curr_state_index,i] = best_prev_state
             
#finding best sequence
        
    state_sequence = []
    last_state = np.argmax(V[:,-1])
    state_sequence.append(last_state)
    for i in range(num_observations-1,0,-1):        
        curr_state = T[state_sequence[-1],i]
        state_sequence.append(curr_state)
    return state_sequence[::-1]       
        
# class structure
class Viterbi():
    
    def __init__(self,init_proba,trans_mat,emission_mat):
        self.init_probability = init_proba
        self.trans_mat = trans_mat
        self.emission_matrix = emission_mat
        #making sure sizes of inputs are reasonable
        assert len(init_proba) == trans_mat.shape[0]
        assert trans_mat.shape[0] == trans_mat.shape[1]
        assert emission_mat.shape[0] == len(init_proba)
        
    def run(self,observations):
        sequence_of_states = ViterbiCalc(self.init_probability,self.emission_matrix,observations,self.trans_mat)
        return sequence_of_states

if __name__ == '__main__':    
 #5
    pi = np.array([[0.04, 0.02, 0.06, 0.04, 0.11, 0.11, 0.01, 0.09, 0.03, 0.05, 0.06, 0.11, 0.05, 0.11, 0.03, 0.08]]).T
    trans = np.array([ \
        [0.08, 0.02, 0.10, 0.05, 0.07, 0.08, 0.07, 0.04, 0.08, 0.10, 0.07, 0.02, 0.01, 0.10, 0.09, 0.01], \
        [0.06, 0.10, 0.11, 0.01, 0.04, 0.11, 0.04, 0.07, 0.08, 0.10, 0.08, 0.02, 0.09, 0.05, 0.02, 0.02], \
        [0.08, 0.07, 0.08, 0.07, 0.01, 0.03, 0.10, 0.02, 0.07, 0.03, 0.06, 0.08, 0.03, 0.10, 0.10, 0.08], \
        [0.08, 0.04, 0.04, 0.05, 0.07, 0.08, 0.01, 0.08, 0.10, 0.07, 0.11, 0.01, 0.05, 0.04, 0.11, 0.06], \
        [0.03, 0.03, 0.08, 0.10, 0.11, 0.04, 0.06, 0.03, 0.03, 0.08, 0.03, 0.07, 0.10, 0.11, 0.07, 0.03], \
        [0.02, 0.05, 0.01, 0.09, 0.05, 0.09, 0.05, 0.12, 0.09, 0.07, 0.01, 0.07, 0.05, 0.05, 0.11, 0.06], \
        [0.11, 0.05, 0.10, 0.07, 0.01, 0.08, 0.05, 0.03, 0.03, 0.10, 0.01, 0.10, 0.08, 0.09, 0.07, 0.02], \
        [0.03, 0.02, 0.16, 0.01, 0.05, 0.01, 0.14, 0.14, 0.02, 0.05, 0.01, 0.09, 0.07, 0.14, 0.03, 0.01], \
        [0.01, 0.09, 0.13, 0.01, 0.02, 0.04, 0.05, 0.03, 0.10, 0.05, 0.06, 0.06, 0.11, 0.06, 0.03, 0.14], \
        [0.09, 0.03, 0.04, 0.05, 0.04, 0.03, 0.12, 0.04, 0.07, 0.02, 0.07, 0.10, 0.11, 0.03, 0.06, 0.09], \
        [0.09, 0.04, 0.06, 0.06, 0.05, 0.07, 0.05, 0.01, 0.05, 0.10, 0.04, 0.08, 0.05, 0.08, 0.08, 0.10], \
        [0.07, 0.06, 0.01, 0.07, 0.06, 0.09, 0.01, 0.06, 0.07, 0.07, 0.08, 0.06, 0.01, 0.11, 0.09, 0.05], \
        [0.03, 0.04, 0.06, 0.06, 0.06, 0.05, 0.02, 0.10, 0.11, 0.07, 0.09, 0.05, 0.05, 0.05, 0.11, 0.08], \
        [0.04, 0.03, 0.04, 0.09, 0.10, 0.09, 0.08, 0.06, 0.04, 0.07, 0.09, 0.02, 0.05, 0.08, 0.04, 0.09], \
        [0.05, 0.07, 0.02, 0.08, 0.06, 0.08, 0.05, 0.05, 0.07, 0.06, 0.10, 0.07, 0.03, 0.05, 0.06, 0.10], \
        [0.11, 0.03, 0.02, 0.11, 0.11, 0.01, 0.02, 0.08, 0.05, 0.08, 0.11, 0.03, 0.02, 0.10, 0.01, 0.11]])
    obs = np.array([[0.01,0.99], \
                    [0.58,0.42], \
                    [0.48,0.52], \
                    [0.58,0.42], \
                    [0.37,0.63], \
                    [0.33,0.67], \
                    [0.51,0.49], \
                    [0.28,0.72], \
                    [0.35,0.65], \
                    [0.61,0.39], \
                    [0.97,0.03], \
                    [0.87,0.13], \
                    [0.46,0.54], \
                    [0.55,0.45], \
                    [0.23,0.77], \
                    [0.76,0.24]])
    data = [[0, 0, 0, 0, 0, 0, 1, 0, 1, 1], [1, 1, 0, 0, 1, 1, 1, 0, 0, 0]]   
                     
    myViterbi = Viterbi(pi,trans,obs) 
    seq0 = myViterbi.run(data[0])
    seq1 = myViterbi.run(data[1])
           