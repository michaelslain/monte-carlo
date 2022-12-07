import numpy as np
import random

# sample data
sampleComments = [
    'this is a good comment',
    'this is a bad comment',
    'this is a bad comment',
    'this is a bad comment',
    'this is a bad comment',
    'this is a horrible comment',
    'this is a horrible comment'
]
sampleLikeProbabilities = [0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01]
sampleDislikeProbabilities = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

# global variables
repeat = 10000
usersPerSample = 100

# store simulation data in memory
simulationMatrix1 = np.full(shape=(repeat, len(sampleComments)), fill_value=0)
simulationMatrix2 = np.full(shape=(repeat, len(sampleComments)), fill_value=0)


# "random" descision based on probability
def decision(probability):
    return random.random() < probability


# simulations
for simulationNum in range(0, repeat):
    for commentNum in range(0, len(sampleComments)):
        likeAmount = 0
        dislikeAmount = 0

        for userNum in range(0, usersPerSample):
            # emulate user liking
            if (decision(sampleLikeProbabilities[commentNum])):
                likeAmount += 1
            # emulate user disliking
            if (decision(sampleDislikeProbabilities[commentNum])):
                dislikeAmount += 1

        # fill data for simulation without dislike
        simulationMatrix1[simulationNum, commentNum] = likeAmount
        # fill data for simulation with dislike
        simulationMatrix2[simulationNum,
                          commentNum] = likeAmount - dislikeAmount

# judge accuracy
points1 = np.zeros(repeat)
points2 = np.zeros(repeat)

for simulationNum in range(0, repeat):
    for i in [0, 1]:
        simulation = []
        if (i == 0):
            simulation = simulationMatrix1[simulationNum]
        else:
            simulation = simulationMatrix2[simulationNum]

        points = 0
        sortedList = np.sort(np.copy(simulation))

        for commentNum in range(0, len(simulation)):
            if sortedList[commentNum] != simulation[commentNum]:
                points -= 1

        if (i == 0):
            points1[simulationNum] = points
            continue

        points2[simulationNum] = points

# compare
points1Avg = np.mean(points1)
points2Avg = np.mean(points2)

# print data
print('-- Average points per type of simulation --')
print('-- (negative is worse)')
print(f'-- Without dislikes: {points1Avg}')
print(f'-- With dislikes: {points2Avg}')
print('--')

if points1Avg < points2Avg:
    print('-- hypothesis is correct, dislikes are better')
else:
    print('-- hypothesis is incorrect, dislikes are worse')
