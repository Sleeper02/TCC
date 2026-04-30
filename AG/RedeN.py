import numpy as np

class RedeN:
    def __init__(self, n_entrada, n_hidden, n_saidas):
        self.peso1 = np.random.randn(n_entrada, n_hidden)
        self.bias1 = np.random.randn(1, n_hidden)
        self.peso2 = np.random.randn(n_hidden, n_saidas)
        self.bias2 = np.random.randn(1, n_saidas)

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        # Versão numericamente estável que evita overflow
        x = np.clip(x, -500, 500)  # Limita valores extremos
        return 1 / (1 + np.exp(-x))

    def prever(self, inputs):
        inputs = np.array(inputs).reshape(1, -1)
        
        hidden = np.dot(inputs, self.peso1) + self.bias1
        hidden = self.relu(hidden)
        
        output = np.dot(hidden, self.peso2) + self.bias2
        output = self.sigmoid(output)
        
        return output[0][0] > 0.5
    
    def prever_multi(self, inputs):
        # Para redes com múltiplas saídas

        inputs = np.array(inputs).reshape(1, -1)

        hidden = np.dot(inputs, self.peso1) + self.bias1
        hidden = self.relu(hidden)

        output = np.dot(hidden, self.peso2) + self.bias2
        output = self.sigmoid(output)
        
        return int(np.argmax(output[0]))   # 0, 1, 2 ou 3