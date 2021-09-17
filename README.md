## Project structure
This is a Q-Learning library that let's you train and evaluate an agent.
The library is in `reinforcementlearing.py`  
The example usage is in `reinforcementlearing_example_usage.py`  
## The example problem
The problem in the example is 'finding next in a sequence'. This is a toy problem and can be solved
by much easier algorithms. We're using it here just to test the correctness of the Q-Learning library.  

For instance let's say we have this sequence:  N N N R N R N N R N R R.  

Knowning 3 items, the problem is to find next item.
For instance after the first N N N, next item is R. After N N R (The 1st N is the 2nd N in the sequence), next is N.  
To solve this problem, we model it a as reinforcement learning one :  
`S={all triplets made from {N,R}}`  
`A={N, R}`

For instance if we are at state (N N R) and apply action N, there are 2 possible next states:
(N R N) or (N R R).    
Actually, each action is a guess. So in the example, if the outcome is (N R N) then we guessed right (Indeed our guessed item (N) and the actual next item are the same).


