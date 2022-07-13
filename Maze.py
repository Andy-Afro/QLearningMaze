import time
import numpy as np
from collections import namedtuple
import random
import pygame

"""
-States
    - -1 = cat
    - 0 = empty
    - 1 = goal
"""
pygame.init()
WHITE = (200, 200, 200)
BLACK = (0, 0, 0)
WIN = pygame.display.set_mode((513, 513))
WIN.fill(WHITE)

MOUSE = pygame.image.load("imgs\\mouse.jpg")
TILE = pygame.image.load("imgs\\tile.png")
CAT = pygame.image.load("imgs\\cat.jpg")
CHEESE = pygame.image.load("imgs\\cheese.jpg")



class Agent:
    def __init__(self, i=0, j=0):
        self.i = i
        self.j = j

    @property
    def loc(self):
        return(self.i, self.j)

    def vmove(self, direction):
        direction = 1 if direction > 0 else -1
        return Agent(self.i + direction, self.j)

    def hmove(self, direction):
        direction = 1 if direction > 0 else -1
        return Agent(self.i, self.j + direction)

    def __repr__(self):
        return str(self.loc)


class QLearning:
    def __init__(self, num_states, num_actions, lr=0.1, discount_factor=1.0):
        self.q = np.zeros((num_states, num_actions))
        self.a = lr
        self.g = discount_factor

    def update(self, st, at, rt, st1):
        q = self.q
        self.q[st, at] = (1 - self.a) * self.q[st, at] + self.a * (rt + self.g * np.max(self.q[st1]))

class Maze:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.env = np.zeros((rows, cols))
        self.mousy = Agent(0, 0)
        self.win = WIN
        self.win.fill(WHITE)
        self.q = np.zeros((rows * cols, 4))
        self.score = 0

    def agent_state(self, a):
        nr, nc = self.env.shape
        return a.i * self.rows + a.j

    def draw(self):
        block = 30
        for x in range(0, 120, block):
            for y in range(0, 120, block):
                rect = pygame.Rect(x, y, block, block)
                pygame.draw.rect(self.win, BLACK, rect, 1)

        for i in range(self.rows):
            for j in range(self.cols):
                if self.env[i, j] == -1:
                    self.place(CAT, j, i)
                elif self.env[i, j] == 0:
                    self.place(TILE, j, i)
                elif self.env[i, j] == 1:
                    self.place(CHEESE, j, i)
                elif self.env[i, j] == 6:
                    self.place(MOUSE, j, i)

    def visualize(self):
        assert self.agent_in_bounds(self.mousy), "out of bounds"
        e = self.env.copy()
        m = self.mousy
        e[m.i, m.j] = 6
        self.draw()

    def in_bounds(self, i, j):
        nr, nc = self.env.shape
        return i >= 0 and i < nr and j >= 0 and j < nc

    def agent_in_bounds(self, a):
        return self.in_bounds(a.i, a.j)

    def is_valid_new_agent(self, a):
        return self.agent_in_bounds(a)

    @property
    def all_actions(self):
        a = self.mousy
        return [
            a.vmove(-1),
            a.vmove(1),
            a.hmove(-1),
            a.hmove(1)
        ]

    def compute_possible_moves(self):
        a = self.mousy
        moves = self.all_actions
        return [(m, ii) for ii, m in enumerate(moves) if self.agent_in_bounds(m)]

    def do_a_move(self, a):
        assert self.is_valid_new_agent(a), "cant go there"
        self.mousy = a
        if self.has_won():
            return 100
        elif self.has_lost():
            return -100
        else:
            return -0.1

    def has_won(self):
        a = self.mousy
        return self.env[a.i, a.j] == 1

    def has_lost(self):
        a = self.mousy
        return self.env[a.i, a.j] == -1

    def place(self, animal, x, y):
        self.win.blit(animal, (x * 30, y * 30))



def make_maze():
    m = Maze(17, 17)
    e = m.env
    e[0, 0] = 6
    e[16, 16] = 1
    m.draw()
    pygame.display.update()
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    run = False
                if event.key == pygame.K_ESCAPE:
                    run = False
                    pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = [x for x in pygame.mouse.get_pos()]
                pos[0] = int(pos[0] / 30)
                pos[1] = int(pos[1] / 30)
                state = e[pos[1], pos[0]]
                if state == 0:
                    e[pos[1], pos[0]] = -1
                    m.place(CAT, pos[0], pos[1])
                elif state == -1:
                    e[pos[1], pos[0]] = 0
                    m.place(TILE, pos[0], pos[1])
                pygame.display.update()

    return m


def make_test_maze():
    m = Maze(4, 4)
    e = m.env
    e[3, 3] = 1
    e[0, 1:2] = -1
    e[1, 2:3] = -1
    e[2, 0] = -1
    e[3, 0:2] = -1
    m.draw()
    return m

def make_big_maze():
    m = Maze(17, 17)
    e = m.env
    e[0, 8] = -1
    e[0, 11:14] = -1
    e[1, 8:10] = -1
    e[1, 13] = -1
    e[1, 16] = -1
    e[2, 3] = -1
    e[2, 6:9] = -1
    e[2, 16] = -1
    e[3, 0:4] = -1
    e[3, 6] = -1
    e[3, 8:10] = -1
    e[3, 12:] = -1
    e[4, 2] = -1
    e[4, 8] = -1
    e[4, 13] = -1
    e[5, 2] = -1
    e[5, 4] = -1
    e[5, 13] = -1
    e[5, 15:] = -1
    e[6, 1:3] = -1
    e[6, 4:7] = -1
    e[6, 9:14] = -1
    e[7, 2] = -1
    e[7, 9] = -1
    e[7, 13] = -1
    e[7, 15] = -1
    e[8, 9] = -1
    e[8, 15] = -1
    e[9, :3] = -1
    e[9, 5:7] = -1
    e[9, 9] = -1
    e[9, 11: 16] = -1
    e[10, 1] = -1
    e[10, 5] = -1
    e[10, 11] = -1
    e[11, 3:9] = -1
    e[11, 11] = -1
    e[11, 14:] = -1
    e[12, 3] = -1
    e[12, 8] = -1
    e[12, 11] = -1
    e[13, :5] = -1
    e[13, 13:16] = -1
    e[14, 10] = -1
    e[14, 15] = -1
    e[15, 1:3] = -1
    e[15, 4:8] = -1
    e[15, 10:13] = -1
    e[15, 15] = -1
    e[16, 1] = -1
    e[16, 4] = -1
    e[16, 12] = -1
    e[16, 15] = -1
    e[16, 16] = 1
    m.draw()
    return m

def reset():
    m = make_big_maze()
    m.place(MOUSE, m.mousy.j, m.mousy.i)
    pygame.display.update()
    return m

if __name__ == "__main__":
    m = make_maze()
    rows = m.rows
    cols = m.cols
    env = m.env
    env[0, 0] = 0
    q = QLearning(m.rows * m.cols, 4)
    clock = pygame.time.Clock()
    pygame.display.update()
    print("STARTING IN")
    for i in range(5, 0, -1):
        print(i)
        time.sleep(1)
    tries = 0
    wait = False
    step_track = []
    run = True
    for i in range(1000):
        if not run:
            break
        m = Maze(rows, cols)
        m.env = env
        m.place(MOUSE, m.mousy.j, m.mousy.i)
        pygame.display.update()
        # time.sleep(1)
        lost = False
        score = 0
        steps = 0
        while not m.has_won() and run:
            for event in pygame.event.get():
                if event == pygame.QUIT:
                    run = False
                    pygame.quit()
            if wait:
                clock.tick(20)
            pygame.display.update()
            if m.has_lost():
                score = 0
                tries += 1
                m.place(CAT, m.mousy.j, m.mousy.i)
                m.mousy.j = 0
                m.mousy.i = 0
                m.place(MOUSE, m.mousy.j, m.mousy.i)
            else:
                steps += 1
                moves = m.compute_possible_moves()
                s = m.agent_state(m.mousy)
                matrix = q.q[s]
                move_idx = np.argmax(matrix)
                move = m.all_actions[move_idx]
                while not m.is_valid_new_agent(move):
                    matrix[move_idx] = -999
                    move_idx = np.argmax(matrix)
                    move = m.all_actions[move_idx]
                at = move_idx
                st = m.agent_state(m.mousy)
                score = m.do_a_move(move)
                rt = score
                st1 = m.agent_state(m.mousy)
                m.score += score


                q.update(st, at, rt, st1)
                # print(f"did move {move_idx}")

                m.visualize()
                m.place(MOUSE, m.mousy.j, m.mousy.i)


        #this is a change
        # positives = 0
        # x, y = q.q.shape
        # for num1 in range(x):
        #     for num2 in range(y):
        #         if q.q[num1, num2] > 0:
        #             positives += 1
        print(f"trial: {i + 1}, num steps {steps}")
        if not wait:
            step_track.append(steps)
            cnt = step_track.count(step_track[0])
            if cnt != len(step_track):
                step_track = []
            elif cnt == 10:
                wait = True
    #
    # m = reset()
    # pygame.display.update()
    # print("STARTING!!!")
    # time.sleep(5)
    # while not m.has_won():
    #     clock.tick(3)
    #     pygame.display.update()
    #     s = m.agent_state(m.mousy)
    #     a_idx = np.argmax(q.q[s])
    #     m.do_a_move(m.all_actions[a_idx])
    #     m.visualize()
    #     m.place(MOUSE, m.mousy.j, m.mousy.i)

    # m = make_maze()
    # print("break")
    # pygame.display.update()
    print(wait)
    # time.sleep(10)
    # print("tries:", tries)
    pygame.quit()
    print("done")