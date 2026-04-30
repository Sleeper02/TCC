import numpy as np
import copy
from RedeN import RedeN

class AlgG:
    def __init__(self, tamanho_populacao=100, taxa_mutacao=0.2):
        self.tamanho_populacao = tamanho_populacao
        self.taxa_mutacao = taxa_mutacao

    def create_populacao(self, n_entradas, n_hidden, n_saidas):
        """
        Cria a primeira geração de cérebros com pesos aleatórios.
        """
        populacao = []

        for _ in range(self.tamanho_populacao):

            cerebro = RedeN(n_entradas, n_hidden, n_saidas)
            populacao.append(cerebro)

        return populacao

    def cal_fitness(self, fitnesses_brutos):
        """
        Calculo do fitness, ou seja, transformamos as pontuações do jogo em probabilidades.
        """
        minimo = min(fitnesses_brutos)
        fitnesses_ajustados = [f - minimo for f in fitnesses_brutos]

        soma = sum(fitnesses_ajustados)
        if soma == 0:
            # Se todos bateram no primeiro cano, chance igual para todos
            return [1.0 / len(fitnesses_brutos) for _ in fitnesses_brutos]
        
        # Transforma a pontuação numa porcentagem (0 a 1)
        probabilidades = [f / soma for f in fitnesses_ajustados]
        return probabilidades

    def selecao(self, populacao, probabilidades):
        """
        Escolhe baseado na probabilidade
        """
        indice_escolhido = np.random.choice(len(populacao), p=probabilidades)
        return populacao[indice_escolhido]

    def crossover(self, pai1, pai2):
        """
        Descendentes terão características genéticas de ambos os escolhidos
        """
        #Garantir que não vai alterar o pai
        filho = copy.deepcopy(pai1) 
        
        # Matriz de verdadeiro/falso aleatória.
        mascara1 = np.random.rand(*pai1.peso1.shape) > 0.5 #caso seja > 0.5, o peso do filho vai ser do pai1, caso contrário, do pai2
        filho.peso1 = np.where(mascara1, pai1.peso1, pai2.peso1) #aq ele utiliza a mascara para escolher os pesos 
        
        mascara_bias1 = np.random.rand(*pai1.bias1.shape) > 0.5
        filho.bias1 = np.where(mascara_bias1, pai1.bias1, pai2.bias1)
        
        mascara2 = np.random.rand(*pai1.peso2.shape) > 0.5
        filho.peso2 = np.where(mascara2, pai1.peso2, pai2.peso2)
        
        mascara_bias2 = np.random.rand(*pai1.bias2.shape) > 0.5
        filho.bias2 = np.where(mascara_bias2, pai1.bias2, pai2.bias2)
        
        return filho

    def mutacao(self, filho):
        """
        Adiciona um  ruído (valores aleatórios) aos pesos.
        """
        mascara_mutacao1 = np.random.rand(*filho.peso1.shape) < self.taxa_mutacao
        ruido1 = np.random.randn(*filho.peso1.shape) * 0.1 # Ruído gaussiano
        filho.peso1 += mascara_mutacao1 * ruido1

        mascara_mutacao2 = np.random.rand(*filho.peso2.shape) < self.taxa_mutacao
        ruido2 = np.random.randn(*filho.peso2.shape) * 0.1
        filho.peso2 += mascara_mutacao2 * ruido2
        
        return filho

    def create_geracao(self, populacao_antiga, fitnesses_brutos):
        """
        Cria a nova geração de cérebros com base na população antiga e suas pontuações.
        """
        probabilidades = self.cal_fitness(fitnesses_brutos)
        nova_populacao = []

        # Elitismo: Mantém o melhor indivíduo intacto para não perder a melhor solução
        n_elite = 5

        melhor_indice = np.argsort(fitnesses_brutos)[-n_elite:] #Pegar os 5 maiores

        for indice in melhor_indice:
            melhor_passaro = copy.deepcopy(populacao_antiga[indice])
            nova_populacao.append(melhor_passaro)
        

        # Gera o restante
        for _ in range(self.tamanho_populacao - n_elite):
            pai1 = self.selecao(populacao_antiga, probabilidades)
            pai2 = self.selecao(populacao_antiga, probabilidades)
            
            filho = self.crossover(pai1, pai2)
            filho_mutante = self.mutacao(filho)
            
            nova_populacao.append(filho_mutante)

        return nova_populacao