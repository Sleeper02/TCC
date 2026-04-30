class Bird:
    def __init__(self, cerebro):
        self.x = 100
        self.y = 100
        self.v_vertical = 0 #v=velocidade
        self.vivo = True
        self.fitness = 0 # Quantos frames o pássaro sobreviveu
        self.cerebro = cerebro

    def pular(self):
        self.v_vertical = -10

    def mover(self, gravity):
        if self.v_vertical <= 5:
            self.v_vertical += gravity / 15
        self.y += self.v_vertical
        self.fitness += 1 # Ganha 1 ponto fitness por cada frame