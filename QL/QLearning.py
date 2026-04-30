import numpy as np
import random
import pickle
import glob # mostra os arquivos para o usuario escolher
import os
import matplotlib.pyplot as plt
from Cobra.visualsnake import VisualSnake # env
from Cobra.snake_no_visual import LearnSnake # env
from Flappy.Flap_no_visual import FlappyNoVisual # env
from Flappy.Flap_visual import VisualFlappy # env


class QLearning:
    def __init__(
        self,
        state_shape: tuple,      # Formato da tabela
        n_actions: int,          
        learning_rate: float = 0.1,   # Alpha
        discount_rate: float = 0.95,  # Gamma
        eps_inicial: float = 1.0,     # Epsilon inicial (exploração)
        eps_minimo: float = 0.01,     # Epsilon mínimo
        eps_descount: float = 0.9998,    # Fator de decaimento do epsilon
    ):
        self.lr = learning_rate
        self.gamma = discount_rate
        self.eps = eps_inicial
        self.eps_min = eps_minimo
        self.eps_decay = eps_descount
        self.n_actions = n_actions

        self.table = np.zeros(state_shape + (n_actions,))
        self.historico_recompensas = []

    def get_action(self, state: tuple) -> int:
        if random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.table[state]))

    #"Bellman Equation" 
    def update(self, state: tuple, action: int, reward: float, new_state: tuple):
        q_atual = self.table[state][action]
        q_futuro = max(self.table[new_state])
        q_novo = (1 - self.lr) * q_atual + self.lr * (reward + self.gamma * q_futuro) #Eq do slide
        self.table[state][action] = q_novo

    def decay_epsilon(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


    def train(
        self,
        env,              # Função que cria um novo ambiente: () -> env
        num_episodes: int = 1, #
        log: int = 25,      # Imprime progresso a cada N episódios (no caso 25)
        save: int = 250,    # Salva modelo a cada N episódios
        save_dir: str = "pickle", # Pasta onde salvar os checkpoints
        starting_episode: int = 0,
    ):
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_dir)
        os.makedirs(save_dir, exist_ok=True) # Garante que a pasta de salvamento exista
        scores = []

        for i in range(starting_episode, starting_episode + num_episodes):
            game_env = env()
            state = game_env.get_state()
            done = False
            total_reward = 0

            #"Atualiza" os estados e o Q
            while not done:
                action = self.get_action(state)
                new_state, reward, done = game_env.step(action)

                self.update(state, action, reward, new_state)
                state = new_state
                total_reward += reward

            self.decay_epsilon()
            self.historico_recompensas.append(total_reward)
            scores.append(total_reward)

            if i % log == 0:
                media = np.mean(self.historico_recompensas[-log:])
                print(f"Episódio {i:6d} | Média recompensa: {media:7.2f} | eps: {self.eps:.4f}")

            if i % save == 0:
                path = os.path.join(save_dir, f"modelo_{i}.pickle")
                self.save(path)
                print(f"  → Modelo salvo em '{path}'")

        print("\n Treinamento concluído!")
        self.plotar_grafico()

    def plotar_grafico(self):
    
        if not self.historico_recompensas:
            return

        plt.figure(figsize=(10, 6))

        janela = 100
        medias_moveis = []
        for i in range(len(self.historico_recompensas)):
            if i < janela:
                medias_moveis.append(np.mean(self.historico_recompensas[:i+1]))
            else:
                medias_moveis.append(np.mean(self.historico_recompensas[i-janela+1:i+1]))

        plt.plot(self.historico_recompensas, label='Recompensa do Episódio', color='lightblue', alpha=0.4)
        plt.plot(medias_moveis, label=f'Média Móvel ({janela} ep.)', color='blue', linewidth=2)

        plt.title('Evolução do Aprendizado - Q-Learning', fontsize=14)
        plt.xlabel('Episódio (Total cumulativo)', fontsize=12)
        plt.ylabel('Recompensa Total', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(fontsize=12)

        plt.savefig('grafico_qlearning_tcc.png', dpi=300, bbox_inches='tight')
        print("✓ Gráfico de evolução salvo como 'grafico_qlearning_tcc.png'")
        plt.show()


    def play(self, env, num_episodes: int = 25, render: bool = True):
        
        #Roda o agente sem exploração  para avaliar
        
        eps_backup = self.eps
        self.eps = 0.0

        for ep in range(num_episodes):
            env_game = env()
            state = env_game.get_state()
            done = False
            total_reward = 0

            while not done:
                action = self.get_action(state) #pega a ação com base na tabela
                state, reward, done = env_game.step(action)
                total_reward += reward
                if render and hasattr(env_game, "render"):
                    env_game.render()

                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env_game.quit()
                        return


            print(f"Episódio {ep + 1} | Recompensa total: {total_reward:.1f}")

        self.eps = eps_backup

    def save(self, path: str):
        dados_para_salvar = {
            "q_table": self.table,
            "historico_recompensas": self.historico_recompensas
        }
        with open(path, "wb") as f:
            pickle.dump(dados_para_salvar, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            dados_carregados = pickle.load(f)
            
            # Se for o formato novo com dicionário
            if isinstance(dados_carregados, dict):
                self.table = dados_carregados["q_table"]
                self.historico_recompensas = dados_carregados.get("historico_recompensas", [])
            else:
                # ficheiro antigo apenas com a tabela
                self.table = dados_carregados
        print(f"✓ Modelo carregado de '{path}'")

    @classmethod #Esse classmethod é para criar um agente sem precisar passar o estado e ações, já que ele vai pegar isso do checkpoint
    def from_checkpoint(cls, path: str, state_shape: tuple, n_actions: int, **kwargs):
        import re #pegar o número do episódio a partir do arquivo
        match = re.search(r"(\d+)", os.path.basename(path))
        starting_ep = int(match.group(1)) if match else 0 #usa como ep inicial

        agent = cls(state_shape, n_actions, **kwargs)
        agent.load(path)

        # Ajusta epsilon para o progresso já feito
        agent.eps = max(agent.eps_min, 1.0 * (agent.eps_decay ** starting_ep))
        return agent, starting_ep


def cli(
    env = FlappyNoVisual, #VisualSnake, #LearnSnake, #FlappyNoVisual,     #VisualFlappy,  # Substitui pela função do ambiente
    state_shape=(15, 7, 15, 15), #(2,2,2,2,2,2,2,2,2,2,2,2),(snake) (15, 7, 15, 15)(flappy)
    n_actions=2,
    pickle_dir: str = os.path.join(os.path.dirname(__file__), "pickle"),
    **agent_args,
):

    op = int(input(
        "Digite sua escolha:\n"
        "1 - Treinar do zero\n"
        "2 - Apenas testar/visualizar\n"
        "3 - Continuar treinamento de modelo existente\n> "
    ))

    if op == 1:
        agent = QLearning(state_shape, n_actions, **agent_args)
        agent.train(env, save_dir=pickle_dir)

    elif op == 2:
        modelos = glob.glob(os.path.join(pickle_dir, "*.pickle"))
        for i, m in enumerate(modelos):
            print(f"{i} - {m}")
        escolha = int(input("Número do modelo: "))
        agent = QLearning(state_shape, n_actions, **agent_args)
        agent.load(modelos[escolha])
        agent.play(env)

    elif op == 3:
        modelos = glob.glob(os.path.join(pickle_dir, "*.pickle"))
        for i, m in enumerate(modelos):
            print(f"{i} - {m}")
        escolha = int(input("Número do modelo para continuar: "))
        agent, start_ep = QLearning.from_checkpoint(
            modelos[escolha], state_shape, n_actions, **agent_args
        )
        agent.train(env, save_dir=pickle_dir, starting_episode=start_ep)

    else:
        print("Opção inválida.")


if __name__ == "__main__":
    cli()