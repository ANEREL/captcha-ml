#!/usr/bin/env python
# coding: utf-8

# # Playing Frozen Lake with Q-learning

# Winter is here. You and your friends were tossing around a frisbee at the park when you made a wild throw that left the frisbee out in the middle of the lake. The water is mostly frozen, but there are a few holes where the ice has melted. If you step into one of those holes, you'll fall into the freezing water. 
# 
# At this time, there's an international frisbee shortage, so it's absolutely imperative that you navigate across the lake and retrieve the disc. However, the ice is slippery, so you won't always move in the direction you intend.
# 
# The surface is described using a grid like the following:
# 
# SFFF       
# FHFH       
# FFFH       
# HFFG
# 
# S: starting point, safe  
# F: frozen surface, safe  
# H: hole, fall to your doom  
# G: goal, where the frisbee is located
# 
# The episode ends when you reach the goal or fall in a hole.  
# You receive a reward of 1 if you reach the goal, and 0 otherwise.
# 
# https://gym.openai.com/envs/FrozenLake-v0/

# In[1]:


import numpy as np
import gym
import random
import time
#from IPython.display import clear_output


# In[2]:


env = gym.make("FrozenLake-v0")


# In[3]:


action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))
print(q_table)


# In[4]:


assert state_space_size == 16
assert action_space_size == 4


# In[5]:


num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001


# In[6]:


rewards_all_episodes = []

# Q-learning algorithm
for episode in range(num_episodes):
    
    print(episode)
    
    state = env.reset()
    done = False
    rewards_current_episode = 0
    
    for step in range(max_steps_per_episode):       
        
        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:]) 
        else:
            action = env.action_space.sample()
        
        new_state, reward, done, info = env.step(action)
        
        # Update Q-table for Q(s,a)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
        
        state = new_state
        rewards_current_episode += reward        
        
        if done == True: 
            break
           
    # Exploration rate decay
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)    
    
    rewards_all_episodes.append(rewards_current_episode)

# Calculate and print the average reward per thousand episodes
rewards_per_thosand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000
print("********Average reward per thousand episodes********\n")
for r in rewards_per_thosand_episodes:
    avg_reward = sum(r/1000)
    print(count, ": ", str(avg_reward))
    count += 1000    

# Print updated Q-table
print("\n\n********Q-table********\n")
print(q_table)


# In[7]:


assert avg_reward > 0.6


# In[8]:


# Watch our agent play Frozen Lake by playing the best action 
# from each state according to the Q-table.
# Run for more episodes to watch longer.

for episode in range(1):
    state = env.reset()
    done = False
    print("*****EPISODE ", episode+1, "*****\n\n\n\n")
    time.sleep(1)

    for step in range(max_steps_per_episode):        
        #clear_output(wait=True)
        env.render()
        time.sleep(0.3)
        
        action = np.argmax(q_table[state,:])        
        new_state, reward, done, info = env.step(action)
        
        if done:
            #clear_output(wait=True)
            env.render()
            if reward == 1:
                print("****You reached the goal!****")
                time.sleep(3)
            else:
                print("****You fell through a hole!****")
                time.sleep(3)
            #clear_output(wait=True)
            break
            
        state = new_state
        
env.close()
