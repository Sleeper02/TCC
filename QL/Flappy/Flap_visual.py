import pygame as pg
import numpy as np
import pickle
import time
import os
from .Flap_no_visual import FlappyNoVisual, STATE_SHAPE, N_ACTIONS

class VisualFlappy(FlappyNoVisual):

    def __init__(self):

        pg.init()
        pg.font.init()

        self.white  = (255, 255, 255)
        self.black  = (  0,   0,   0)
        self.orange = (255, 165,   0)

        self.window = pg.display.set_mode((1280, 720))
        pg.display.set_caption("Flappy Bird — Q-Learning")

        self.font    = pg.font.SysFont("Courier New", 50, bold=True)
        self.font_sm = pg.font.SysFont("Courier New", 28, bold=True)
        self.clock   = pg.time.Clock()

        assets = os.path.dirname(os.path.abspath(__file__))
        background = pg.image.load(os.path.join(assets, 'Background.png'))
        bird       = pg.image.load(os.path.join(assets, 'Bird.png'))
        ground     = pg.image.load(os.path.join(assets, 'Ground.png'))
        pipe       = pg.image.load(os.path.join(assets, 'Pipe.png'))
        pipe_usd   = pg.image.load(os.path.join(assets, 'Pipe Up Side Down.png'))

        self.img_background = pg.transform.scale(background, (2120, 634))
        self.img_bird       = pg.transform.scale(bird,       (  51,  36))
        self.img_ground     = pg.transform.scale(ground,     (1010,  86))
        self.img_pipe       = pg.transform.scale(pipe,       ( 123, 600))
        self.img_pipe_usd   = pg.transform.scale(pipe_usd,   ( 123, 600))

        self.background_1_pos = [   0,   0]
        self.background_2_pos = [2120,   0]
        self.ground_1_pos     = [   0, 634]
        self.ground_2_pos     = [1010, 634]
        self.ground_3_pos     = [2020, 634]

        super().__init__()  

    def _move_pipes(self):

        
        super()._move_pipes()

        self.background_1_pos[0] -= 1.2
        self.background_2_pos[0] -= 1.2
        if self.background_1_pos[0] <= -2120:
            self.background_1_pos[0] = 0
            self.background_2_pos[0] = 2120

        self.ground_1_pos[0] -= 1.2
        self.ground_2_pos[0] -= 1.2
        self.ground_3_pos[0] -= 1.2
        if self.ground_1_pos[0] <= -1010:
            self.ground_1_pos[0] = 0
            self.ground_2_pos[0] = 1010
            self.ground_3_pos[0] = 2020

    def render(self):
        self.draw()

    def draw(self, episode=None, avg_score=None):

        self.window.blit(self.img_background, self.background_1_pos)
        self.window.blit(self.img_background, self.background_2_pos)


        for pipe in [self.pipe_1_pos, self.pipe_2_pos, self.pipe_3_pos,
                     self.pipe_4_pos, self.pipe_5_pos]:
            self.window.blit(self.img_pipe,     (pipe[0], pipe[1]))
            self.window.blit(self.img_pipe_usd, (pipe[0], pipe[1] - 800))

        # Chão
        self.window.blit(self.img_ground, self.ground_1_pos)
        self.window.blit(self.img_ground, self.ground_2_pos)
        self.window.blit(self.img_ground, self.ground_3_pos)

        # Pássaro
        self.window.blit(self.img_bird, (int(self.bird_x), int(self.bird_y)))


        border = 5
        x, y, w, h = 1100, 50, 150, 100
        text = self.font.render(str(self.score), 1, self.white)
        text_x = x + (w / 2) - (text.get_width() / 2)
        text_y = y + (h / 2) - (text.get_height() / 2)
        pg.draw.rect(self.window, self.orange, (x, y, w, h))
        pg.draw.rect(self.window, self.black,  (x, y, w, h), border)
        pg.draw.rect(self.window, self.white,  (x+border, y+border, w-border*2, h-border*2), border)
        self.window.blit(text, (text_x, text_y))

        # HUD — episódio e média
        if episode is not None:
            ep_surf = self.font_sm.render(f"Episódio: {episode}", 1, self.white)
            self.window.blit(ep_surf, (20, 20))
        if avg_score is not None:
            avg_surf = self.font_sm.render(f"Média: {avg_score:.1f}", 1, self.white)
            self.window.blit(avg_surf, (20, 60))

        pg.display.update()
        self.clock.tick(60)

    def check_quit(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return True
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                return True
        return False

    def quit(self):
        pg.quit()


def run_game(model_path):
    
    #visualização usando um modelo salvo.
    with open(model_path, 'rb') as f:
        q_table = pickle.load(f)

    print(f"Modelo carregado: {model_path}")
    print("Pressione ESC ou feche a janela para parar.")

    scores      = []
    game_number = 1

    while True:
        print(f"\n=== PARTIDA {game_number} ===")

        game  = VisualFlappy()
        state = game.get_state()
        done  = False
        steps_sem_ponto = 0
        max_steps = 5000

        while not done and steps_sem_ponto < max_steps:
            action = int(np.argmax(q_table[state]))

            old_score = game.score
            state, reward, done = game.step(action)

            avg = np.mean(scores) if scores else 0
            game.draw(episode=game_number, avg_score=avg)

            if game.score > old_score:
                steps_sem_ponto = 0
            else:
                steps_sem_ponto += 1

            if game.check_quit():
                game.quit()
                print(f"\nInterrompido após {game_number} partidas.")
                if scores:
                    print(f"Score médio : {np.mean(scores):.2f}")
                    print(f"Melhor score: {max(scores)}")
                return scores

        scores.append(game.score)
        print(f"Score da partida {game_number}: {game.score}")
        if len(scores) >= 10:
            print(f"Média últimas 10: {np.mean(scores[-10:]):.2f}")

        game.quit()
        time.sleep(0.4)
        game_number += 1


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_game(sys.argv[1])
    else:
        print("Execute: python visual_flappy.py caminho_do_modelo.pickle")