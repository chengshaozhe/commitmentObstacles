import os
DIRNAME = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import unittest
from ddt import ddt, data, unpack
from dataAnalysis.dataAnalysis import calculateAvoidCommitmnetZoneAll, calculateAvoidCommitmentRatio, calculateFirstOutZoneRatio, calculateFirstIntentionRatio, calculateFirstIntention, calculateFirstIntentionConsistency, inferGoal


@ddt
class TestAnalysisFunctions(unittest.TestCase):
    @data(((4, 4), (13, 8), (9, 12), [(4, 7), (6, 6), (5, 6), (4, 8), (9, 8), (7, 7), (9, 4), (8, 5), (5, 8), (6, 7), (5, 5), (7, 6), (8, 6), (9, 7), (6, 4), (5, 4), (4, 5), (7, 5), (8, 7), (9, 6), (6, 5), (4, 6), (6, 8), (5, 7), (7, 4), (8, 8), (9, 5), (7, 8), (8, 4)])
          )
    @unpack
    def testCalculateAvoidCommitmnetZoneAll(self, playerGrid, target1, target2, groundTruthZone):
        zone = calculateAvoidCommitmnetZoneAll(playerGrid, target1, target2)
        truthValue = np.array_equal(zone, groundTruthZone)
        self.assertTrue(truthValue)

    @data(([(1, 2), [3, 2], [4, 2], [5, 5], [5, 2]], [(3, 2), (4, 2), (5, 2), (5, 3), (5, 4)], 0.75)
          )
    @unpack
    def testCalculateAvoidCommitmentRatio(self, trajectory, zone, groundTruthRatio):
        avoidCommitmentRatio = calculateAvoidCommitmentRatio(trajectory, zone)
        truthValue = np.array_equal(avoidCommitmentRatio, groundTruthRatio)
        self.assertTrue(truthValue)

    @data(([(1, 2), [3, 2], [4, 2], [5, 5], [5, 2]], [(3, 2), (4, 2), (5, 2), (5, 3), (5, 4)], 0.5))
    @unpack
    def testCalculateFirstOutZoneRatio(self, trajectory, zone, groundTruthRatio):
        avoidCommitmentRatio = calculateFirstOutZoneRatio(trajectory, zone)
        truthValue = np.array_equal(avoidCommitmentRatio, groundTruthRatio)
        self.assertTrue(truthValue)

    @data(([0, 0, 0, 0, 2, 0, 0, 2, 2, 2], 0.5),
          ([0, 0, 0, 0, 1, 1], 5 / 6))
    @unpack
    def testCalculateFirstIntentionRatio(self, goalList, groundTruthRatio):
        avoidCommitmentRatio = calculateFirstIntentionRatio(goalList)
        truthValue = np.array_equal(avoidCommitmentRatio, groundTruthRatio)
        self.assertTrue(truthValue)

    @data(([0, 0, 0, 0, 2, 0, 0, 2, 2, 2], 2),
          ([0, 0, 0, 0, 1, 2, 0, 0, 2, 2, 2], 1),
          ([0, 0, 0, 0, 0, 0, 0, 0], 0))
    @unpack
    def testCalculateFirstIntention(self, goalList, groundTruthGoal):
        firstIntention = calculateFirstIntention(goalList)
        truthValue = np.array_equal(firstIntention, groundTruthGoal)
        self.assertTrue(truthValue)

    @data(([0, 0, 0, 0, 2, 0, 0, 2, 2, 2], 1),
          ([0, 0, 0, 0, 1, 2, 0, 0, 2, 2, 0], 0),
          ([0, 0, 0, 0, 0, 0, 0, 0], 1))
    @unpack
    def testCalculateFirstIntentionConsistency(self, goalList, groundTruthConsis):
        firstIntention = calculateFirstIntentionConsistency(goalList)
        truthValue = np.array_equal(firstIntention, groundTruthConsis)
        self.assertTrue(truthValue)

    @data(((0, 0), (0, 1), (2, 2), (3, 3), 0),
          ((0, 0), (0, 1), (0, 2), (3, 0), 1),
          ((0, 0), (0, 1), (2, 0), (0, 3), 2))
    @unpack
    def testInferGoal(self, originGrid, aimGrid, targetGridA, targetGridB, groundTruthGoal):
        inferredGoal = inferGoal(originGrid, aimGrid, targetGridA, targetGridB)
        truthValue = np.array_equal(inferredGoal, groundTruthGoal)
        self.assertTrue(truthValue)


if __name__ == '__main__':
    unittest.main()
