'''
Q-learning approach for different RL problems
as part of the basic series on reinforcement learning @
https://github.com/vmayoral/basic_reinforcement_learning

Inspired by https://gym.openai.com/evaluations/eval_kWknKOkPQ7izrixdhriurA

        @author: Victor Mayoral Vilches <victor@erlerobotics.com>
'''
import pickle
import random
import rospkg
import utils

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('turtlebot3_rl_sim')
result_outdir = pkg_path + '/src/results/qlearn'
model_outdir = pkg_path + '/src/models/qlearn/discrete_no_greedy'


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = utils.load_q(model_outdir + '/qlearn_qtable_ep3000.txt')
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha  # discount constant
        self.gamma = gamma  # discount factor
        self.actions = actions
        self.count_same = 0
        self.count_diff = 0

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        '''
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
            self.count_same += 1
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)
            self.count_diff += 1

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q);
            mag = max(abs(minQ), abs(maxQ))

            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))]
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]
        if return_q:  # if they want it, give it!
            return action, q

        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma * maxqnew)

    def get_qtable(self):
        return self.q

    # Test phase purposes
    def save_q(self, qtable, outdir, name):
        with open(str(outdir) + '/' + name + '.txt', 'wb') as f:
            pickle.dump(qtable, f)

    def load_q(self, file):
        q = None
        with open(file, 'rb') as f:
            q = pickle.loads(f.read())

        return q

    def set_q(self, new_q):
        self.q = new_q
