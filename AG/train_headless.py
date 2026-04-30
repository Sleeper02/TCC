import sys
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Flappy'))
from AlgG import AlgG
from Flappy.Bird import Bird


class FlapBirdsHeadless:
    """Versão do jogo sem pygame (roda 10x+ mais rápido)"""
    def __init__(self):
        self.gravity = 5
        self.bird_pos = [100, 100]
        self.score = 0
        self.last_random_height_for_pipe = 0
        self.bird_passing_through_obstacle = False
        
        self.pipe_1_pos = [400, self.new_height_for_pipe()]
        self.pipe_2_pos = [800, self.new_height_for_pipe()]
        self.pipe_3_pos = [1200, self.new_height_for_pipe()]
        self.pipe_4_pos = [1600, self.new_height_for_pipe()]
        self.pipe_5_pos = [2000, self.new_height_for_pipe()]
        
        self.birds = []
        self.birds_mortos = []
        self.frame_count = 0
        self.max_frames = 10000

    def new_height_for_pipe(self):
        new_height = np.random.randint(6, 11) * 50
        while self.last_random_height_for_pipe == new_height:
            new_height = np.random.randint(6, 11) * 50
        self.last_random_height_for_pipe = new_height
        return new_height

    def movement(self):
        # Pipes
        for pipe in [self.pipe_1_pos, self.pipe_2_pos, self.pipe_3_pos, self.pipe_4_pos, self.pipe_5_pos]:
            pipe[0] -= 1.2
        
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

        # Birds
        for bird in self.birds:
            if bird.vivo:
                distancia_x = self.pipe_1_pos[0] - bird.x
                
                if distancia_x < -130:
                    distancia_x = self.pipe_2_pos[0] - bird.x
                    distancia_yB = self.pipe_2_pos[1] - 400
                else:
                    distancia_yB = self.pipe_1_pos[1] - 400

                diferenca_y = bird.y - distancia_yB

                inputs = [
                    bird.y / 720.0,
                    bird.v_vertical / 15.0,
                    distancia_x / 1280.0,
                    diferenca_y / 720.0
                ]

                pulo = bird.cerebro.prever(inputs)
                
                if pulo:
                    bird.pular()
                
                bird.mover(self.gravity)

    def collision(self):
        for bird in self.birds:
            if bird.vivo:
                if bird.y + 36 > 634 or bird.y < 0:
                    bird.vivo = False
                
                if self.pipe_1_pos[0] < bird.x + 51 and self.pipe_1_pos[0] + 123 > bird.x:
                    if bird.y < self.pipe_1_pos[1] - 200 or bird.y + 36 > self.pipe_1_pos[1]:
                        bird.vivo = False
                
                if not bird.vivo:
                    self.birds_mortos.append(bird)
        
        self.birds = [b for b in self.birds if b.vivo]

    def update(self):
        self.movement()
        self.collision()
        self.frame_count += 1


class Trainer:
    def __init__(self, pop_size=100, generations=50):
        self.ag = AlgG(tamanho_populacao=pop_size, taxa_mutacao=0.05)
        self.pop_size = pop_size
        self.generations = generations
        self.melhores = []
        self.medias = []

    def rodar_jogo(self, birds):
        """Roda um jogo completo"""
        jogo = FlapBirdsHeadless()
        jogo.birds = birds
        
        while jogo.birds and jogo.frame_count < jogo.max_frames:
            jogo.update()
        
        return jogo.birds_mortos

    def treinar(self):
        print(f"Treinamento HEADLESS (sem display)")
        print(f"População: {self.pop_size} | Gerações: {self.generations}\n")
        
        modelo_path = 'modelos_treinados.pkl'
        if os.path.exists(modelo_path):
            print("✓ Carregando modelos treinados anteriores para continuar a evolução...")
            with open(modelo_path, 'rb') as f:
                dados_carregados = pickle.load(f)

                if isinstance(dados_carregados, dict):
                    cerebros = dados_carregados["cerebros"]
                    self.melhores = dados_carregados.get("historico_melhores", [])
                    self.medias = dados_carregados.get("historico_medias", [])
                else: 
                    # Retrocompatibilidade se abrir um modelo antigo
                    cerebros = dados_carregados
        else:

            print("⚠ Nenhum modelo anterior encontrado. Começando do zero (Geração 1)...")
            cerebros = self.ag.create_populacao(4, 6, 1)
            
        birds = [Bird(cerebro) for cerebro in cerebros]
        
        for gen in range(self.generations):
            # Roda o jogo
            birds_mortos = self.rodar_jogo(birds)
            
            # Stats
            fitnesses = [b.fitness for b in birds_mortos]
            melhor = max(fitnesses) if fitnesses else 0
            media = np.mean(fitnesses) if fitnesses else 0
            
            self.melhores.append(melhor)
            self.medias.append(media)
            
            print(f"Gen {gen+1:3d} | Melhor: {melhor:5.0f} | Média: {media:6.1f} | Pássaros: {len(birds_mortos)}")
            
            # Evolui
            cerebros_novos = self.ag.create_geracao([b.cerebro for b in birds_mortos], fitnesses)
            birds = [Bird(c) for c in cerebros_novos]
        
        print("\n✓ Treinamento concluído!")

        plt.figure(figsize=(10, 6))
        
        # Desenha as linhas
        plt.plot(range(1, len(self.melhores) + 1), self.melhores, label='Melhor Aptidão', linewidth=2)
        plt.plot(range(1, len(self.medias) + 1), self.medias, label='Aptidão Média', linestyle='--', linewidth=2)
        
        
        plt.title('Evolução do Algoritmo Genético - Flappy Bird', fontsize=14)
        plt.xlabel('Geração', fontsize=12)
        plt.ylabel('Fitness (Frames Sobrevividos)', fontsize=12)
        
       
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(fontsize=12)
        
        plt.savefig('grafico_evolucao_tcc.png', dpi=300, bbox_inches='tight')
        plt.show()

        return birds


if __name__ == "__main__":
    trainer = Trainer(pop_size=100, generations=1000)
    birds_finais = trainer.treinar()
    
    # Salva os cérebros treinados
    cerebros = [b.cerebro for b in birds_finais]

    dados_para_salvar = {
        "cerebros": cerebros,
        "historico_melhores": trainer.melhores,
        "historico_medias": trainer.medias
    }

    with open('modelos_treinados.pkl', 'wb') as f:
        pickle.dump(dados_para_salvar, f)
    
    print("\n✓ Modelos salvos em 'modelos_treinados.pkl'")
    print("Agora você pode usar no Flap_Birds.py!")
