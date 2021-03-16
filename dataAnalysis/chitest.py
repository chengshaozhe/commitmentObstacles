import numpy as np
from scipy import stats
obs = [11, 9]
exp = [10, 10]


# obs = [752, 448]
# exp = [600, 600]

# obs = [56, 100]
# exp = [np.mean(obs)] * len(obs)

# obs = [839, 846, 833, 782, 747]
# exp = [840] * len(obs)

chisquare, p = stats.chisquare(obs, f_exp=exp)
print('p:', p)
print('chisquare:', chisquare)


def calCramerV(chisquare, sampleSize):
    CramerV = np.sqrt(chisquare / sampleSize)
    return CramerV


sampleSize = np.sum(obs)
CramerV = calCramerV(chisquare, sampleSize)
print('CramerV:', CramerV)



# χ2 (1) = 16.2, P < 0.001, VCramer = 0.9


