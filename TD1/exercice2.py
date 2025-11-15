import numpy as np

int_to_char = {
    0 : 'u',
    1 : 'r',
    2 : 'd',
    3 : 'l'
}

policy_one_step_look_ahead = {
    0 : [-1,0],
    1 : [0,1],
    2 : [1,0],
    3 : [0,-1]
}

def policy_int_to_char(pi,n):

    pi_char = ['']
    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0 or i == n-1 and j == n-1:
                continue
            pi_char.append(int_to_char[pi[i,j]])
    pi_char.append('')
    return np.asarray(pi_char).reshape(n,n)


# 1- Policy evaluation
def policy_evaluation(n, pi, v, Gamma, threshhold): 

  v_new = v.copy()
  while True:
      delta = 0
      v_old = v_new.copy()  
      for i in range(n):
          for j in range(n):
              if (i == 0 and j == 0) or (i == n - 1 and j == n - 1):
                  continue              
              action = pi[i, j]
              R = -1
              di, dj = policy_one_step_look_ahead[action]
              ni, nj = i + di, j + dj
              if not (0 <= ni < n and 0 <= nj < n):
                  ni, nj = i, j  
              if (ni == 0 and nj == 0) or (ni == n - 1 and nj == n - 1):
                  V_successor = 0.0
              else:
                  V_successor = v_old[ni, nj]
              v_new[i, j] = R + Gamma * V_successor              
              delta = max(delta, abs(v_new[i, j] - v_old[i, j]))
      if delta < threshhold:
          break    
  return v_new


# 2- Policy improvement
def policy_improvement(n, pi, v, Gamma):
  
  pi_stable = True
  new_pi = pi.copy() 
  for i in range(n):
      for j in range(n):
          if (i == 0 and j == 0) or (i == n - 1 and j == n - 1):
              continue
          old_action = pi[i, j]          
          q_values = np.zeros(4)
          
          for action in range(4): 
              R = -1 
              di, dj = policy_one_step_look_ahead[action]
              ni, nj = i + di, j + dj
              if not (0 <= ni < n and 0 <= nj < n):
                  ni, nj = i, j  
              if (ni == 0 and nj == 0) or (ni == n - 1 and nj == n - 1):
                  V_successor = 0.0
              else:
                  V_successor = v[ni, nj]              
              q_values[action] = R + Gamma * V_successor
          best_action = np.argmax(q_values)
          new_pi[i, j] = best_action          
          if old_action != best_action:
              pi_stable = False
              
  return new_pi, pi_stable


# 3- Policy initialization
def policy_initialization(n):

  pi = np.random.randint(0, 4, size=(n, n), dtype=int)
  pi[0, 0] = -1 
  pi[n - 1, n - 1] = -1
  return pi


# 4- Policy Iteration algorithm
def policy_iteration(n,Gamma,threshhold):

    pi = policy_initialization(n=n)
    v = np.zeros(shape=(n,n))
    while True:
        v = policy_evaluation(n=n,v=v,pi=pi,threshhold=threshhold,Gamma=Gamma)
        pi , pi_stable = policy_improvement(n=n,pi=pi,v=v,Gamma=Gamma)
        if pi_stable:
            break
    return pi , v


# Main Code to Test
n = 4
Gamma = [0.8,0.9,1]
threshhold = 1e-4
for _gamma in Gamma:
    pi , v = policy_iteration(n=n,Gamma=_gamma,threshhold=threshhold)
    pi_char = policy_int_to_char(n=n,pi=pi)
    print()
    print("Gamma = ",_gamma)
    print()
    print(pi_char)
    print()
    print()
    print(v)
