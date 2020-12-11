import numpy as np

class FacilityLocationCIFAR:

    def __init__(self, V, D=None, fnpy=None):
        if D is not None:
          self.D = D
        else:
          self.D = np.load(fnpy)

        self.D *= -1
        self.D -= self.D.min()
        self.V = V
        self.curVal = 0
        self.gains = []
        self.curr_max = np.zeros_like(self.D[0])

    def inc(self, sset, ndx):
        if len(sset + [ndx]) > 1:
            new_dists = np.stack([self.curr_max, self.D[ndx]], axis=0)
            return new_dists.max(axis=0).sum()
        else:
            return self.D[sset + [ndx]].sum()

    def add(self, sset, ndx, delta):
        self.curVal += delta
        self.gains += delta,
        self.curr_max = np.stack([self.curr_max, self.D[ndx]], axis=0).max(axis=0)
        return self.curVal

        cur_old = self.curVal
        if len(sset + [ndx]) > 1:
            self.curVal = self.D[:, sset + [ndx]].max(axis=1).sum()
        else:
            self.curVal = self.D[:, sset + [ndx]].sum()
        self.gains.extend([self.curVal - cur_old])
        return self.curVal
