"""
  python snake_ag.py          -> treina headless e salva modelos
  python snake_ag.py visual   -> visualiza o melhor modelo salvo
"""

import sys
import os
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AlgG import AlgG
from RedeN import RedeN


class SnakeEnv:
    """
    Ambiente headless da cobra.
    """
    SCREEN_W = 600
    SCREEN_H = 400
    CELL    = 10
    COLS    = SCREEN_W // CELL   # 60
    ROWS    = SCREEN_H // CELL   # 40

    def __init__(self, cerebro: RedeN):
        self.cerebro = cerebro
        self.reset()

    def reset(self):
        self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self.r = self.ROWS // 2
        self.c = self.COLS // 2
        self.dir = "right"
        self.dr, self.dc = 0, 1
        self.coords = [(self.r, self.c)]
        self.board[self.r][self.c] = 1
        self.length = 1
        self.alive = True
        self.frames = 0
        self.food_eaten = 0
        self.frames_sem_comida = 0
        self.food_r, self.food_c = self._gerar_comida()
        self.board[self.food_r][self.food_c] = 2

    def _gerar_comida(self):
        while True:
            fr = random.randint(0, self.ROWS - 1)
            fc = random.randint(0, self.COLS - 1)
            if self.board[fr][fc] == 0:
                return fr, fc

    def _valido(self, r, c):
        return 0 <= r < self.ROWS and 0 <= c < self.COLS

    def _perigoso(self, r, c):
        if not self._valido(r, c):
            return 1
        return 1 if self.board[r][c] == 1 else 0

    # ── estado (12 entradas) ──
    def get_state(self):
        hr, hc = self.coords[-1]
        return [
            float(self.dir == "left"),
            float(self.dir == "right"),
            float(self.dir == "up"),
            float(self.dir == "down"),
            float(self.food_r < hr),
            float(self.food_r > hr),
            float(self.food_c < hc),
            float(self.food_c > hc),
            float(self._perigoso(hr + 1, hc)),
            float(self._perigoso(hr - 1, hc)),
            float(self._perigoso(hr, hc + 1)),
            float(self._perigoso(hr, hc - 1)),
        ]

    def step(self):
        state = self.get_state()
        acao = self.cerebro.prever_multi(state)     # 0=esq 1=dir 2=cima 3=baixo
        dirs = [("left", 0, -1), ("right", 0, 1), ("up", -1, 0), ("down", 1, 0)]
        nome, dr, dc = dirs[acao]

        # Impede inversão de sentido
        opostos = {"left": "right", "right": "left", "up": "down", "down": "up"}
        if nome != opostos.get(self.dir, ""):
            self.dir, self.dr, self.dc = nome, dr, dc

        nr = self.coords[-1][0] + self.dr
        nc = self.coords[-1][1] + self.dc

        # Morte por parede ou corpo
        if not self._valido(nr, nc) or self.board[nr][nc] == 1:
            self.alive = False
            return

        self.board[nr][nc] = 1
        self.coords.append((nr, nc))

        # Remove cauda se não cresceu
        if len(self.coords) > self.length:
            tr, tc = self.coords.pop(0)
            self.board[tr][tc] = 0

        # Comeu?
        if nr == self.food_r and nc == self.food_c:
            self.length += 1
            self.food_eaten += 1
            self.frames_sem_comida = 0
            self.food_r, self.food_c = self._gerar_comida()
            self.board[self.food_r][self.food_c] = 2
        else:
            self.frames_sem_comida += 1

        # Mata por loop: ficou muito tempo sem comer
        if self.frames_sem_comida > 200:
            self.alive = False

        self.frames += 1

    @property  
    def fitness(self):
        hr, hc = self.coords[-1]
        distancia_comida = abs(hr - self.food_r) + abs(hc - self.food_c)
        return self.food_eaten * 500 - self.frames_sem_comida * 3 - distancia_comida

    def run(self, max_frames=5000):
        """Roda um episódio completo."""
        while self.alive and self.frames < max_frames:
            self.step()



class Trainer:
    N_ENTRADAS = 12
    N_HIDDEN   = 16
    N_SAIDAS   = 4

    def __init__(self, pop=100, geracoes=100, taxa_mutacao=0.08):
        self.ag = AlgG(tamanho_populacao=pop, taxa_mutacao=taxa_mutacao)
        self.pop = pop
        self.geracoes = geracoes
        self.modelo_path = "snake_ag_modelo.pkl"

    def _avaliar(self, redes):
        """Roda cada cobra e retorna lista de fitness."""
        resultados = []
        for rede in redes:
            env = SnakeEnv(rede)
            env.run()
            resultados.append((env.fitness, env.food_eaten))
        return resultados

    def treinar(self, continuar=False):
        print("=" * 50)
        print("  Snake com Algoritmo Genético")
        print(f"  Pop: {self.pop} | Gerações: {self.geracoes}")
        print("=" * 50)

        melhor_historico = []
        media_historico = []

        # Carrega ou cria população inicial
        if continuar and os.path.exists(self.modelo_path):
            with open(self.modelo_path, "rb") as f:
                dados_carregados = pickle.load(f)
                
                # Se for o modelo novo (dicionário)
                if isinstance(dados_carregados, dict):
                    redes = dados_carregados["cerebros"]
                    melhor_historico = dados_carregados.get("historico_melhores", [])
                    media_historico = dados_carregados.get("historico_medias", [])

                else: # Se for o modelo antigo (apenas lista)
                    redes = dados_carregados
            print(f"✓ Modelos carregados de '{self.modelo_path}'")
        else:
            redes = self.ag.create_populacao(self.N_ENTRADAS, self.N_HIDDEN, self.N_SAIDAS)
            print("⚠ Iniciando do zero...")

        for gen in range(1, self.geracoes + 1):
            resultados = self._avaliar(redes)

            fitnesses     = [r[0] for r in resultados]
            comidas       = [r[1] for r in resultados]
            melhor_fit   = max(fitnesses)
            media_fit    = np.mean(fitnesses)
            melhor_comida = max(comidas)

            melhor_historico.append(melhor_fit)
            media_historico.append(media_fit)

            print(f"Gen {gen:4d} | Melhor fitness: {melhor_fit:7.0f} | "
                  f"Média: {media_fit:7.1f} | Melhor comida: {melhor_comida:3d}")

            # Evolui
            redes = self.ag.create_geracao(redes, fitnesses)

        
            if gen % 10 == 0:
                dados_para_salvar = {
                    "cerebros": redes,
                    "historico_melhores": melhor_historico,
                    "historico_medias": media_historico
                }
                with open(self.modelo_path, "wb") as f:
                    pickle.dump(dados_para_salvar, f)
                print(f"  ✓ Salvo em '{self.modelo_path}'")
        
        dados_para_salvar = {
            "cerebros": redes,
            "historico_melhores": melhor_historico,
            "historico_medias": media_historico
        }

        with open(self.modelo_path, "wb") as f:
            pickle.dump(dados_para_salvar, f)
        print(f"\n✓ Treinamento concluído! Modelo salvo em '{self.modelo_path}'")

        plt.figure(figsize=(10, 6))

        plt.plot(range(1, len(melhor_historico) + 1), melhor_historico, label='Melhor Aptidão', linewidth=2, color='green')
        plt.plot(range(1, len(media_historico) + 1), media_historico, label='Aptidão Média', linestyle='--', linewidth=2, color='orange')
        
        plt.title('Evolução do Algoritmo Genético - Snake', fontsize=14)
        plt.xlabel('Geração (Total cumulativo)', fontsize=12)
        plt.ylabel('Fitness', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(fontsize=12)
        
        plt.savefig('grafico_snake_ag_tcc.png', dpi=300, bbox_inches='tight')
        print("✓ Gráfico de evolução salvo como 'grafico_snake_ag_tcc.png'")
        
        return redes


def visualizar(modelo_path="snake_ag_modelo.pkl", geracao_zero=False):
    try:
        import pygame
    except ImportError:
        print("pygame não instalado. Execute: pip install pygame")
        return
    
    if geracao_zero:
        print("Cobra aleatória (Geração 0)...")
        ag_temp = AlgG(tamanho_populacao=1, taxa_mutacao=0.0)
        # Entradas: 12, Ocultas: 16, Saídas: 4 (mesmo tamanho do seu Trainer)
        redes = ag_temp.create_populacao(12, 16, 4) 
        rede = redes[0]

    else:
        if not os.path.exists(modelo_path):
            print(f"Modelo '{modelo_path}' não encontrado.")
            return

        with open(modelo_path, "rb") as f:
            dados_carregados = pickle.load(f)
            
            if isinstance(dados_carregados, dict):
                redes = dados_carregados["cerebros"]
                print(f"Modelo encontrado.")

            else:
                redes = dados_carregados

        # Usa o primeiro da lista (elite)
        rede = redes[0]

    pygame.init()
    CELL = 10
    W, H = 600, 400
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Snake — Algoritmo Genético")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    small = pygame.font.Font(None, 24)

    BLACK  = (  0,   0,   0)
    WHITE  = (255, 255, 255)
    GREEN  = ( 50, 200,  50)
    DKGREEN= ( 20, 120,  20)
    RED    = (220,  50,  50)
    BLUE   = ( 50, 150, 255)

    partida = 0
    scores  = []

    while True:
        partida += 1
        env = SnakeEnv(rede)

        while env.alive:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return

            env.step()

            # Desenho
            screen.fill(BLACK)

            # Grade leve
            for r in range(env.ROWS):
                for c in range(env.COLS):
                    pygame.draw.rect(screen, (15, 15, 15),
                                     (c * CELL, r * CELL, CELL, CELL), 1)

            # Cobra
            for i, (r, c) in enumerate(env.coords):
                cor = BLUE if i == len(env.coords) - 1 else (GREEN if i % 2 == 0 else DKGREEN)
                pygame.draw.rect(screen, cor, (c * CELL, r * CELL, CELL, CELL))

            # Comida
            pygame.draw.rect(screen, RED,
                             (env.food_c * CELL, env.food_r * CELL, CELL, CELL))

            # HUD
            score_txt = font.render(f"Score: {env.food_eaten}", True, WHITE)
            screen.blit(score_txt, (10, 10))

            partida_txt = small.render(f"Partida: {partida}", True, WHITE)
            screen.blit(partida_txt, (10, 50))

            if scores:
                avg_txt = small.render(f"Média: {np.mean(scores):.1f}  "
                                       f"Melhor: {max(scores)}", True, WHITE)
                screen.blit(avg_txt, (10, 70))

            fit_txt = small.render(f"Fitness: {env.fitness}", True, (180, 180, 180))
            screen.blit(fit_txt, (10, 90))

            pygame.display.flip()
            clock.tick(15)

        scores.append(env.food_eaten)
        print(f"Partida {partida} | Score: {env.food_eaten} | "
              f"Fitness: {env.fitness} | Média: {np.mean(scores):.1f}")

        pygame.time.wait(400)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "visual":
        visualizar()
    else:
        op = input(
            "\nEscolha:\n"
            "  1 - Treinar do zero\n"
            "  2 - Continuar treinamento\n"
            "  3 - Visualizar modelo salvo\n"
            "  4 - Visualizar geração 0 (aleatório)\n"
            "  > "
        ).strip()

        if op == "1":
            trainer = Trainer(pop=100, geracoes=50, taxa_mutacao=0.08)
            trainer.treinar(continuar=False)
        elif op == "2":
            trainer = Trainer(pop=100, geracoes=19, taxa_mutacao=0.08)
            trainer.treinar(continuar=True)
        elif op == "3":
            visualizar(geracao_zero=False)
        elif op == "4":
            visualizar(geracao_zero=True)
        else:
            print("Opção inválida.")