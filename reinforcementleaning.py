import numpy as np
import random
from abc import ABC, abstractmethod
from sklearn.metrics import roc_curve, auc, roc_auc_score, log_loss, accuracy_score, confusion_matrix


def get_precision(y_true, y_pred_labels):
    #y_pred_labels = (y_pred>th).astype(int)
    cm = confusion_matrix(y_true, y_pred_labels)
    denominator = cm.sum().sum()
    numerator = np.array([cm[j,j] for j in range(cm.shape[1]) ]).sum()
    precision = round(numerator/denominator, 3)
    return precision

class Env(ABC):
    def __init__(self, n_states):
        self.n_states=n_states
    
    '''
    returns [new state, reward, is episode ended, info]
    '''
    @abstractmethod
    def step(self, a):
        pass
    
    @abstractmethod
    def reset(self):
        pass

class Agent:
    def __init__(self, n_actions, env, qlearning):
        self.n_actions = n_actions
        self.env=env
        env.agent = self
        self.Q=np.zeros((env.n_states, n_actions))  
        self.qlearning = qlearning
        qlearning.env = env
        self.epsilon = qlearning.epsilon
        self.alpha = qlearning.alpha
        self.gamma = qlearning.gamma
        self.train = True
    
    def play(self, s):
        # Exploration-exploitation trade-off
        if(self.train):
            exploration_rate_threshold = random.uniform(0, 1)
        else:
            exploration_rate_threshold = 1
        # Take new a :
        #    exploit
        if exploration_rate_threshold > self.qlearning.epsilon:
            a = np.argmax(self.Q[s,:]) 
        #    explore
        else:
            a = random.randint(0,self.n_actions-1)        
        sp, R, done, info = self.env.step(a)
        if(self.train):
            self.Q[s, a] = self.Q[s, a] * (1 - self.qlearning.alpha) + \
                self.qlearning.alpha * (R + self.qlearning.alpha * np.max(self.Q[sp, :]))    
        # Update Q-table
        # Set new state
        # Add new R 
        return (sp, R, done, info, a)

                
                
                        
class QLearning:   
    def __init__(self, agent , max_steps_per_episode, \
                 epsilon=1, alpha = 0.1, gamma=0.9, epsilon_min=0.01, epsilon_max = 1, exploration_decay_rate = 0.01 ):

        self.agent = agent
        self.max_steps_per_episode = max_steps_per_episode
        self.epsilon = epsilon 
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.exploration_decay_rate = exploration_decay_rate
        '''
        num_episodes = 10000
        max_steps_per_episode = 100
        
        learning_rate = 0.1 -> alpha
        discount_rate = 0.99 -> gamma
        
        exploration_rate = 1 -> epsilon
        max_exploration_rate = 1
        min_exploration_rate = 0.01
        exploration_decay_rate = 0.01 
        '''

    def train(self):
        self.agent.train = True
        # Q-learning algorithm
        for episode in range(self.agent.env.num_episodes):
            s = self.env.reset()
            # initialize new episode params           
            for step in range(self.max_steps_per_episode): 
                sp, R, done, info, a = self.agent.play(s)
                s = sp
                # this episode is ended, start a new one
                if done == True: 
                    break
                
            self.epsilon = self.epsilon_min + \
                (self.epsilon_max - self.epsilon_min) * np.exp(-self.exploration_decay_rate*episode)        
                
    def test(self):
        self.agent.env.num_episodes = len(self.agent.env.X_test)
        self.agent.env.counter = 0
        y_true=[]
        self.agent.train = False
        for episode in range(self.agent.env.num_episodes):
            s = self.env.reset()
            # initialize new episode params           
            for step in range(self.max_steps_per_episode): 
                sp, R, done, info, a = self.agent.play(s)
                s = sp
                y_true.append(1 if info == 'R' else 0)
                # this episode is ended, start a new one
                if done == True: 
                    break   
        y_pred_labels = self.agent.env.y_test
        precision = get_precision(y_true, y_pred_labels) 
        print(precision)