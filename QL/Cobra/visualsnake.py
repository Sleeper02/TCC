import pygame
import random
import numpy as np
import pickle
import time

# Código refeito por IA para visualização do jogo

class VisualSnake:
    def __init__(self):
        pygame.init()
        
        # Configurações do jogo (seguindo padrão do snake_no_visual)
        self.screen_width = 600
        self.screen_height = 400
        self.snake_size = 10
        self.snake_speed = 15
        
        # Configurações do pygame
        self.cell_size = self.snake_size
        self.width = self.screen_width
        self.height = self.screen_height
        
        # Cores
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        
        # Pygame setup
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake Q-Learning")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        self.reset()
    
    def render(self):
        self.draw()

    def reset(self):
        """Reinicia o jogo seguindo padrão do snake_no_visual"""
        self.snake_coords = []
        self.snake_length = 1
        self.dir = "right"
        self.board = np.zeros((self.screen_height // self.snake_size, self.screen_width // self.snake_size))
        
        self.game_close = False
     
        self.x1 = self.screen_width / 2
        self.y1 = self.screen_height / 2
        
        self.r1, self.c1 = self.coords_to_index(self.x1, self.y1)
        self.board[self.r1][self.c1] = 1
             
        self.c_change = 1
        self.r_change = 0
          
        self.food_r, self.food_c = self.generate_food()
        self.board[self.food_r][self.food_c] = 2
        self.survived = 0
        self.steps_sem_comida = 0

        # Executar primeiro step
        self.step_internal()
        
        return self.get_state()
    
    def coords_to_index(self, x, y):
        """Converte coordenadas pixel para índices da matriz"""
        r = int(y // 10)
        c = int(x // 10)
        return (r, c)
    
    def generate_food(self):
        """Gera comida em posição aleatória (seguindo padrão do snake_no_visual)"""
        food_c = int(round(random.randrange(0, self.screen_width - self.snake_size) / 10.0))
        food_r = int(round(random.randrange(0, self.screen_height - self.snake_size) / 10.0))
        if self.board[food_r][food_c] != 0:
            food_r, food_c = self.generate_food()
        return food_r, food_c
    
    def valid_index(self, r, c):
        """Verifica se índices são válidos"""
        return 0 <= r < len(self.board) and 0 <= c < len(self.board[0])
    
    def is_unsafe(self, r, c):
        """Verifica se posição é perigosa (seguindo padrão do snake_no_visual)"""
        if self.valid_index(r, c):
            if self.board[r][c] == 1:
                return 1
            return 0
        else:
            return 1
    
    def get_dist(self, r1, c1, r2, c2):
        """Calcula distância entre dois pontos"""
        return ((r2 - r1) ** 2 + (c2 - c1) ** 2) ** 0.5
    
    def game_over(self):
        """Verifica se o jogo terminou"""
        return self.game_close
    
    def get_state(self):
        """Retorna o estado atual seguindo padrão do snake_no_visual"""
        head_r, head_c = self.snake_coords[-1]
        state = []
        state.append(int(self.dir == "left"))
        state.append(int(self.dir == "right"))
        state.append(int(self.dir == "up"))
        state.append(int(self.dir == "down"))
        state.append(int(self.food_r < head_r))
        state.append(int(self.food_r > head_r))
        state.append(int(self.food_c < head_c))
        state.append(int(self.food_c > head_c))
        state.append(self.is_unsafe(head_r + 1, head_c))
        state.append(self.is_unsafe(head_r - 1, head_c))
        state.append(self.is_unsafe(head_r, head_c + 1))
        state.append(self.is_unsafe(head_r, head_c - 1))
        return tuple(state)
    
    def step_internal(self):
        """Lógica interna de movimento (sem action externa)"""
        return self.step("None")
    
    def step(self, action="None"):
        """Executa ação seguindo padrão do snake_no_visual"""
        if action == "None":
            action = random.choice(["left", "right", "up", "down"])
        else:
            action = ["left", "right", "up", "down"][action]
        
        reward = 0
 
        if action == "left" and (self.dir != "right" or self.snake_length == 1):
            self.c_change = -1
            self.r_change = 0
            self.dir = "left"
        elif action == "right" and (self.dir != "left" or self.snake_length == 1):
            self.c_change = 1
            self.r_change = 0
            self.dir = "right"
        elif action == "up" and (self.dir != "down" or self.snake_length == 1):
            self.r_change = -1
            self.c_change = 0
            self.dir = "up"
        elif action == "down" and (self.dir != "up" or self.snake_length == 1):
            self.r_change = 1
            self.c_change = 0
            self.dir = "down"

        # Verificar limites antes do movimento
        if self.c1 >= self.screen_width // self.snake_size or self.c1 < 0 or self.r1 >= self.screen_height // self.snake_size or self.r1 < 0:
            self.game_close = True
        self.c1 += self.c_change
        self.r1 += self.r_change
        
        self.snake_coords.append((self.r1, self.c1))
        
        if self.valid_index(self.r1, self.c1):
            self.board[self.r1][self.c1] = 1
        
        if len(self.snake_coords) > self.snake_length:
            rd, cd = self.snake_coords[0]
            del self.snake_coords[0]
            if self.valid_index(rd, cd):
                self.board[rd][cd] = 0
 
        for r, c in self.snake_coords[:-1]:
            if r == self.r1 and c == self.c1:
                self.game_close = True
 
        # Verificar se comeu comida
        if self.c1 == self.food_c and self.r1 == self.food_r:
            self.food_r, self.food_c = self.generate_food()
            self.board[self.food_r][self.food_c] = 2
            self.snake_length += 1
            reward = 1
            self.steps_sem_comida = 0 
        else:
            self.steps_sem_comida += 1 

        if self.steps_sem_comida > 200:
            self.game_close = True
            reward = -10
        
        # death = -10 reward
        if self.game_close:
            reward = -10
        self.survived += 1
        
        return self.get_state(), reward, self.game_close
    
    def draw(self, game_number=None, total_games=None, avg_score=None):
        """Desenha o jogo na tela"""
        self.screen.fill(self.BLACK)
        
        # Desenhar cobra usando coordenadas do board
        for i, (r, c) in enumerate(self.snake_coords):
            color = self.BLUE if i == len(self.snake_coords) - 1 else self.GREEN  # Cabeça azul, corpo verde
            x = c * self.snake_size
            y = r * self.snake_size
            rect = pygame.Rect(x, y, self.snake_size, self.snake_size)
            pygame.draw.rect(self.screen, color, rect)
        
        # Desenhar comida
        food_x = self.food_c * self.snake_size
        food_y = self.food_r * self.snake_size
        food_rect = pygame.Rect(food_x, food_y, self.snake_size, self.snake_size)
        pygame.draw.rect(self.screen, self.RED, food_rect)
        
        # Desenhar informações
        score = self.snake_length - 1
        score_text = self.font.render(f"Score: {score}", True, self.WHITE)
        self.screen.blit(score_text, (10, 10))
        
        if game_number is not None and total_games is not None:
            game_text = self.font.render(f"Partida: {game_number}/{total_games}", True, self.WHITE)
            self.screen.blit(game_text, (10, 50))
        
        if avg_score is not None:
            avg_text = self.font.render(f"Score Médio: {avg_score:.1f}", True, self.WHITE)
            self.screen.blit(avg_text, (10, 90))
        
        pygame.display.flip()
        self.clock.tick(15)  # 15 FPS para boa visualização
    
    def check_quit(self):
        """Verifica se o usuário fechou a janela"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False
    
    def quit(self):
        """Fecha o jogo"""
        pygame.quit()
    
    def run_game(self, episode):
        """Executa jogo usando tabela Q salva (compatibilidade com snake_no_visual)"""
        filename = f"pickle/{episode}.pickle"
        with open(filename, 'rb') as file:
            table = pickle.load(file)
        current_length = 2
        steps_unchanged = 0
        while not self.game_over():
            state = self.get_state()
            action = np.argmax(table[state])
            if steps_unchanged == 1000:
                break
            self.step(action)
            if self.snake_length != current_length:
                steps_unchanged = 0
                current_length = self.snake_length
            else:
                steps_unchanged += 1
            
            # Desenhar durante execução
            self.draw()
            if self.check_quit():
                break
                
        return self.snake_length


def run_game(model_path):
    """Executa múltiplas partidas do Snake com visualização"""
    # Carregar o modelo
    with open(model_path, 'rb') as file:
        q_table = pickle.load(file)
    
    print(f"Modelo carregado: {model_path}")
    print("Executando múltiplas partidas...")
    print("Pressione ESC ou feche a janela para parar")
    
    scores = []
    game_number = 1
    
    while True:  # Loop infinito de partidas
        print(f"\n=== PARTIDA {game_number} ===")
        
        game = VisualSnake()
        current_state = game.get_state()
        done = False
        steps_without_food = 0
        max_steps = 1000
        
        while not done and steps_without_food < max_steps:
            # Escolher ação usando o modelo
            action = np.argmax(q_table[current_state])
            
            # Executar ação
            old_score = game.snake_length - 1
            current_state, reward, done = game.step(action)
            
            # Atualizar informações na tela
            avg_score = np.mean(scores) if scores else 0
            game.draw(game_number, "∞", avg_score)
            
            # Contar passos sem comer
            new_score = game.snake_length - 1
            if new_score > old_score:
                steps_without_food = 0
            else:
                steps_without_food += 1
            
            # Verificar se usuário fechou a janela
            if game.check_quit():
                game.quit()
                print(f"\nJogo interrompido pelo usuário após {game_number} partidas")
                if scores:
                    print(f"Score médio: {np.mean(scores):.2f}")
                    print(f"Melhor score: {max(scores)}")
                    print(f"Pior score: {min(scores)}")
                return scores
        
        final_score = game.snake_length - 1
        scores.append(final_score)
        
        print(f"Score da partida {game_number}: {final_score}")
        if len(scores) >= 10:
            print(f"Score médio das últimas 10 partidas: {np.mean(scores[-10:]):.2f}")
        
        game.quit()
        
        # Pequena pausa entre partidas
        time.sleep(0.5)
        
        game_number += 1


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_game(sys.argv[1])
    else:
        print("Execute: python visualsnake.py caminho_do_modelo.pickle")

