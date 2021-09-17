'''
Created on Dec 9, 2019

@author: Ouba
'''

import pandas as pd
import random
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, log_loss, accuracy_score, confusion_matrix
from reinforcement.reinforcementleaning import *

def obtenir_y_pour_une_ligne(row):
    #print("row is : " + str(row))
    return 1 if (row['a']=='R' and row['R']==1) or (row['a']=='N' and row['N']==1) else 0



def get_r(s2a2rs, s,a):
    map = s2a2rs
    for xi in s:
        if xi in map:
            map = map[xi]
        else:
            raise Exception("Aucune recompense observée pour ce couple (s,a)")       
    if a in map:
        list = map[a]
    else:
        raise Exception("Aucune recompense observée pour ce couple (s,a)") 
    # choisir au hasard une recompense
    choix = random.randint(0,len(list)-1)
    return list[choix]


class EnvRoulette(Env):
    def __init__(self, n_states, s0, num_episodes):
        super().__init__(n_states)
        
        
        df = pd.read_csv("JFDATJEU.csv")   
        df['xa'] = df['Etap1'].apply(lambda x: list(x))
        #df["xa2"]= df["Etap1"].str.split(".", n = 4, expand = True)
        for i in range(3):
            df['x'+str(i+1)] = df['xa'].apply(lambda x: x[i])
        df['a'] = df['xa'].apply(lambda x: x[3])

        df['y'] = df.apply(lambda x:obtenir_y_pour_une_ligne(x), axis=1)


        X = df
        y = df['y']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)        
        
        self.num_episodes = len(self.X_train)
        
        s2a2r_raws ={}
        self.actions = ['N','R']
        for index, x in self.X_train.iterrows():
            #s = ['R','R','N']
            #a = ['N']
            s = [x['x' + str(i+1)] for i in range(3)]
            a = x['a']
            r = [x['R'], x['N']]
            map = s2a2r_raws
            for xi in s:
                if xi in map:
                    map = map[xi]
                else:
                    m = {}
                    map[xi] = m
                    map = m        
            if a in map:
                r_raws = map[a]
            else:
                r_raws = []
                map[a] = r_raws
            r_raws.append(x)
        
        self.s2a2r_raws = s2a2r_raws
        self.s = s0
        # créer le decodeur
        i = 0
        decodes = {}
        self.decodes = decodes
        encodes = []
        self.encodes = encodes
        for x1 in self.actions:
            for x2 in self.actions:
                for x3 in self.actions:
                    decodes[(x1,x2,x3)] = i
                    encodes.append([x1,x2,x3])
                    i+=1

                    
    def decode(self, s):
        return self.decodes[tuple(s)]
    def encode(self, s):
        if s>=len(self.encodes):
            v=1
        return self.encodes[s]
        #return self.
    '''
    returns [new state, reward, is episode ended, info]
    '''
    def step(self, a):
        # lire d'Excel ce qui s'est passé 
        s = self.s

        a = self.actions[a]
        s2a2rs =self.s2a2r_raws
        s_ = self.encode(s)
        r = get_r(s2a2rs, s_, a)

        R =-1
    
        if(a=='R'):
            if r['R']==1:
                R = 1
                lastxi = 'R'
            else:
                lastxi = 'N'
        elif(a=='N'):
            if r['N']==1:
                R = 1
                lastxi = 'N'
            else:
                lastxi = 'R'
        sp_ = s_[1:]
        sp_.append(lastxi)
        sp = self.decode(sp_)
        done = True
        info = lastxi
        
        return (sp, R, done, info)
        
    

    def reset(self):  
        if self.agent.train:
            i = random.randint(0,self.n_states-1)
            self.s = i
        else:
            x = self.X_test.iloc[self.counter]
            s_ = [x['x' + str(i+1)] for i in range(3)]
            i = self.decode(s_)
            self.counter = self.counter + 1
        return i


if __name__ == '__main__':
    print("Entrainement")
    n_actions = 2
    num_episodes = None
    s0 = ['R','R','R']
    n_states = 2**3
    env = EnvRoulette(n_states, s0, num_episodes)
    agent = None

    max_steps_per_episode = sys.maxsize
 
    qlearning = QLearning(agent, max_steps_per_episode)
    agent = Agent(n_actions, env, qlearning)
    qlearning.agent = agent
    qlearning.train()
    
    
    # evaluation
    print("Evaluation")
    qlearning.test()
    

    print("Meilleur coup à jouer pour chaque configuration:")
    for s in range(n_states):
        s_ = env.encode(s)
        sp, R, done, info, a = agent.play(s)
        print (str(s_)+" --> " + env.actions[a])

