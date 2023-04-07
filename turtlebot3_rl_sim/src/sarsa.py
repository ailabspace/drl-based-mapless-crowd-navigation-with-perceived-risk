import random
import pickle
import rospkg
import utils

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('turtlebot3_rl_sim')
result_outdir = pkg_path + '/src/results/sarsa'
model_outdir = pkg_path + '/src/models/sarsa/discrete'


class Sarsa:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}  # utils.load_q(model_outdir + '/sarsa_qtable_ep1500.txt')

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions
        self.count_same = 0
        self.count_diff = 0

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
            self.count_same += 1
            # print("NO Q entry: ", oldv)
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)
            self.count_diff += 1
            # print("NEW Q entry: ", oldv)
        # print("COUNT NONE: ", self.count_same)
        # print("COUNT DIFF: ", self.count_diff)

    def chooseAction(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q = [self.getQ(state, a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = self.actions[i]
        # print("NOT MATCHED STATE: ", str(state))
        # print("MATCHING STATES: ", str(self.count))
        return action

    def learn(self, state1, action1, reward, state2, action2):
        qnext = self.getQ(state2, action2)
        self.learnQ(state1, action1, reward, reward + self.gamma * qnext)

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

    def get_action_qtable(self, state):
        q = []
        for action in range(3):  # action 0, 1, 2
            q.append(self.q.get((state, action), 0.0))
        print(q)
        # if state in self.q:
        #     action = self.q.get(state, default=None)
        #     print(action)
        # else:
        #     print("NOT AVAILABLE")
