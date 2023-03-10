import numpy as np

#1
def calculating_forward(observations, states, start_prob, trans_prob, emm_prob, end_st):
    # forward part of the algorithm
    fwd = []
    f_prev = {}
    for i, observation_i in enumerate(observations):
        f_curr = {}
        for st in states:
            if i == 0:
                # base case for the forward part
                prev_f_sum = start_prob[st]
            else:
                prev_f_sum = sum(f_prev[k]*trans_prob[k][st] for k in states)

            f_curr[st] = emm_prob[st][observation_i] * prev_f_sum

        fwd.append(f_curr)
        f_prev = f_curr

    p_fwd = sum(f_curr[k] * trans_prob[k][end_st] for k in states)
    return fwd,p_fwd

#2    
def calculating_backwards(observations, states, start_prob, trans_prob, emm_prob, end_st):
    # backward part of the algorithm
    bkw = []
    b_prev = {}
    for i, observation_i_plus in enumerate(reversed(observations[1:]+(None,))):
        b_curr = {}
        for st in states:
            if i == 0:
                # base case for backward part
                b_curr[st] = trans_prob[st][end_st]
            else:
                b_curr[st] = sum(trans_prob[st][l] * emm_prob[l][observation_i_plus] * b_prev[l] for l in states)

        bkw.insert(0,b_curr)
        b_prev = b_curr

    p_bkw = sum(start_prob[l] * emm_prob[l][observations[0]] * b_curr[l] for l in states)
    return bkw,p_bkw


#3
def forward_backward_algorithm(observations, states, start_prob, trans_prob, emm_prob, end_st):
    fwd,p_fwd = calculating_forward  (observations, states, start_prob, trans_prob, emm_prob, end_st)
    bkw,p_bkw = calculating_backwards(observations, states, start_prob, trans_prob, emm_prob, end_st)
    # merging the two parts
    posterior = []
    for i in range(len(observations)):
        posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st in states})

    assert p_fwd == p_bkw  # shumim zrihim lihiet shavim
    return fwd, bkw, posterior


states = ('Healthy', 'Fever')
end_state = 'E'
 
observations = ('normal', 'cold', 'dizzy')
 
start_probability = {'Healthy': 0.6, 'Fever': 0.4}
 
transition_probability = {
                           'Healthy' : {'Healthy': 0.69, 'Fever': 0.3,  'E': 0.01},
                           'Fever'   : {'Healthy': 0.4,  'Fever': 0.59, 'E': 0.01},
                          }
 
emission_probability = {
                          'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
                          'Fever'   : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
                       }

if __name__ == "__main__":
    fwd, bkw, posterior = forward_backward_algorithm(observations,states,start_probability,transition_probability,emission_probability,end_state)
