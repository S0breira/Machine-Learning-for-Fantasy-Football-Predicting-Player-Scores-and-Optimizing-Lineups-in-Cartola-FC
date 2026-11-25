import math
from collections import defaultdict

PRIOR = {"GOL": -4.0, "ZAG": -1.5, "LAT": -1.0, "MEI": 0.0, "ATA": 0.5}


class Cobrador:
    """Modelo de cobrador de pênalti. Cada atleta tem uma força; a chance de cobrar é dividida
    entre os jogadores em campo por softmax.
    """

    def __init__(self, eta=0.5, lam=0.05):
        """eta é o tamanho do passo e lam a força que puxa cada atleta de volta para o prior da posição."""
        self.eta = eta
        self.lam = lam
        self.forca = {}
        self.pos = {}

    def forca_de(self, jog, pos):
        """Força do atleta; na primeira vez começa no prior da posição."""
        if jog not in self.forca:
            self.forca[jog] = PRIOR.get(pos, 0.0)
            self.pos[jog] = pos
        return self.forca[jog]

    def chances(self, campo):
        """Chance de cada jogador em campo cobrar: exp(força) / soma dos exp(força) dos que estão em campo."""
        f = {j: self.forca_de(j, p) for j, p in campo.items()}
        maior = max(f.values())
        ex = {j: math.exp(v - maior) for j, v in f.items()}
        tot = sum(ex.values())
        return {j: v / tot for j, v in ex.items()}

    def credito(self, ev):
        """Divide um pênalti convertido entre os autores de gol, na proporção (chance de cobrar x gols)."""
        ch = self.chances(ev["campo"])
        w = {j: ch[j] * ev["gols"][j] for j in ev["gols"] if j in ch}
        tot = sum(w.values())
        if tot == 0:
            return {}
        return {j: v / tot for j, v in w.items()}

    def passo(self, ev):
        """Atualiza as forças com um evento: cada atleta em campo recebe eta·(crédito - chance).
        Quem marcou e era provável cobrador sobe; quem não marcou desce. Pênalti perdido mostra
        o cobrador com certeza.
        """
        ch = self.chances(ev["campo"])
        grad = defaultdict(float)
        for _ in range(ev["convertidos"]):
            cr = self.credito(ev)
            for j in ch:
                grad[j] += cr.get(j, 0.0) - ch[j]
        for quem, n in ev["perdidos"].items():
            if quem in ch:
                for j in ch:
                    alvo = 1.0 if j == quem else 0.0
                    grad[j] += n * (alvo - ch[j])
        for j, g in grad.items():
            self.forca[j] += self.eta * g

    def ajustar(self, hist, epocas=5):
        """Repassa todo o histórico (os casos ambíguos antigos são reavaliados com as forças atuais)
        e puxa as forças de volta para o prior da posição.
        """
        for _ in range(epocas):
            for ev in hist:
                self.passo(ev)
            for j in self.forca:
                prior = PRIOR.get(self.pos[j], 0.0)
                self.forca[j] -= self.eta * self.lam * (self.forca[j] - prior)
