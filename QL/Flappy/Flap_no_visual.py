import random
import numpy as np


# Bins = limitar o mundo

BINS_Y      = np.linspace(0,    720,  16)   # 15 faixas
BINS_VEL    = np.linspace(-10,  6,    8)    # 7 faixas
BINS_DIST_X = np.linspace(-130, 1280, 16)  # 16 faixas
BINS_DIFF_Y = np.linspace(-720, 720, 16)  # 16 faixas

STATE_SHAPE = (15, 7, 15, 15)
N_ACTIONS   = 2  # 0 | 1 


def discretizar(valor, bins): # np.digitize retorna o índice do bin onde o valor se encaixa
    return int(np.clip(np.digitize(valor, bins) - 1, 0, len(bins) - 2))


class FlappyNoVisual:

    def __init__(self):

        self.gravity = 5

        self.bird_x = 100
        self.bird_y = 100
        self.bird_w = 51
        self.bird_h = 36
        self.v_vertical = 0

        self.score = 0
        self.last_random_height_for_pipe = 0
        self.bird_passing_through_obstacle = False
        self.game_close = False
        self.survived = 0

        self.pipe_1_pos = [ 400, self.new_height_for_pipe()]
        self.pipe_2_pos = [ 800, self.new_height_for_pipe()]
        self.pipe_3_pos = [1200, self.new_height_for_pipe()]
        self.pipe_4_pos = [1600, self.new_height_for_pipe()]
        self.pipe_5_pos = [2000, self.new_height_for_pipe()]

    def new_height_for_pipe(self):

        new_height = random.randint(6, 10) * 50
        while self.last_random_height_for_pipe == new_height:
            new_height = random.randint(6, 10) * 50
        self.last_random_height_for_pipe = new_height
        return new_height

    def _move_pipes(self):

        self.pipe_1_pos[0] -= 1.2
        self.pipe_2_pos[0] -= 1.2
        self.pipe_3_pos[0] -= 1.2
        self.pipe_4_pos[0] -= 1.2
        self.pipe_5_pos[0] -= 1.2

        if self.pipe_1_pos[0] <= -123:
            self.pipe_1_pos[0] = self.pipe_2_pos[0]
            self.pipe_1_pos[1] = self.pipe_2_pos[1]
            self.pipe_2_pos[0] = self.pipe_3_pos[0]
            self.pipe_2_pos[1] = self.pipe_3_pos[1]
            self.pipe_3_pos[0] = self.pipe_4_pos[0]
            self.pipe_3_pos[1] = self.pipe_4_pos[1]
            self.pipe_4_pos[0] = self.pipe_5_pos[0]
            self.pipe_4_pos[1] = self.pipe_5_pos[1]
            self.pipe_5_pos[0] = 1877
            self.pipe_5_pos[1] = self.new_height_for_pipe()

    def _move_bird(self):

        if self.v_vertical <= 5:
            self.v_vertical += self.gravity / 15
        self.bird_y += self.v_vertical

    def _check_score(self):

        reward = 0
        if self.pipe_1_pos[0] < self.bird_x + self.bird_w and \
           self.pipe_1_pos[0] + 123 > self.bird_x:
            self.bird_passing_through_obstacle = True
        else:
            if self.bird_passing_through_obstacle:
                self.score += 1
                reward = 10  # passou pelo cano
            self.bird_passing_through_obstacle = False
        return reward

    def _check_collision(self):

        if self.bird_y + self.bird_h > 634 or self.bird_y < 0:
            return True

        if self.pipe_1_pos[0] < self.bird_x + self.bird_w and \
           self.pipe_1_pos[0] + 123 > self.bird_x:
            if self.bird_y < self.pipe_1_pos[1] - 200 or \
               self.bird_y + self.bird_h > self.pipe_1_pos[1]:
                return True
        return False

    def get_state(self) -> tuple:

        distancia_x = self.pipe_1_pos[0] - self.bird_x
        if distancia_x < -130:
            distancia_x  = self.pipe_2_pos[0] - self.bird_x
            distancia_yB = self.pipe_2_pos[1] - 400
        else:
            distancia_yB = self.pipe_1_pos[1] - 400

        diferenca_y = self.bird_y - distancia_yB

        state =  ( 
            discretizar(self.bird_y,     BINS_Y),
            discretizar(self.v_vertical, BINS_VEL),
            discretizar(distancia_x,     BINS_DIST_X),
            discretizar(diferenca_y,     BINS_DIFF_Y),
        )

        for i, (idx, shape) in enumerate(zip(state, STATE_SHAPE)):
            if idx >= shape:
                print(f"ERRO no índice {i}: valor={idx} shape={shape}")
                print(f"bird_y={self.bird_y} v_vertical={self.v_vertical} dist_x={distancia_x} diff_y={diferenca_y}")

        return state

    def game_over(self):
        return self.game_close

    def step(self, action: int):
        
        #action: 0 = não pula | 1 = pula
        reward = 0.1  # sobreviveu mais um frame

        if action == 1:
            self.v_vertical = -10 

        self._move_bird()
        self._move_pipes()
        reward += self._check_score()

        if self._check_collision():
            self.game_close = True
            reward = -10

        self.survived += 1
        return self.get_state(), reward, self.game_close