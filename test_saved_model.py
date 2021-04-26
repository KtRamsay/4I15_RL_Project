from collections import namedtuple
import numpy as np
#from tensorboardX import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import copy
import random

from map_generation import resetMap
from observation_functions import getObsSpaceRepresentation
from plot_functions import plotLifespanBar, plotMap, plotObservationInput, plotSuccess, plotSuccessReward

import torch 
import torch.nn as nn
import torch.nn.functional as F


#Define  actor network
class Actor(nn.Module):
    def __init__(self,obSize,hiddenSize, hiddenSize2, numActions):
        super(Actor,self).__init__()
        if useObservationSpace:
            self.conv1 = nn.Conv2d(1,5,5,1).double()
            self.conv2 = nn.Conv2d(5,18,3).double()
            self.conv3 = nn.Conv2d(18,3,3).double()

            self.flatten = nn.Flatten().double()

            #self.fc1 = nn.Linear(86,80).double()
            self.fc1 = nn.Linear(78,80).double()
        else:
            #self.fc1 = nn.Linear(11,80).double()
            self.fc1 = nn.Linear(3,80).double()
        self.fc2 = nn.Linear(80,80).double()
        self.fc3 = nn.Linear(80,numActions).double()

        self.tanh = nn.Tanh()

    def forward(self,spaceMatrix, additionalData):
        if useObservationSpace:
            spaceMatrix = spaceMatrix.view((spaceMatrix.shape[0], 1, spaceMatrix.shape[1], spaceMatrix.shape[1]))

            spaceMatrix = F.relu(self.conv1(spaceMatrix))
            spaceMatrix = F.avg_pool2d(spaceMatrix,2,2)
            spaceMatrix = F.relu(self.conv2(spaceMatrix))  
            spaceMatrix = F.avg_pool2d(spaceMatrix,2,2)
            spaceMatrix = F.relu(self.conv3(spaceMatrix)) 
            spaceMatrix = F.avg_pool2d(spaceMatrix,2,1) 
            observation = self.flatten(spaceMatrix)

            state = torch.cat((observation, additionalData), 1)
        else :
            state = additionalData
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        state = self.tanh(self.fc3(state))
        return state


#Define learning model settings
HIDDEN_SIZE = 128
HIDDEN_SIZE2 = 100
BATCH_SIZE = 64
TARGET_UPDATE = 10
MAX_EPISODE_ITERS = 150
EPISODES = 500
SEE_EPIDODE = EPISODES + 1

#Set if to load a model
setName = "ObservationSpaceTest"
#saveNum = "1619187260"
#saveNum = "1619186452"
#saveNum = "1619203557"
saveNum = "1619265915"

savedActorPath = "Model_Saves/" + setName + "/Actor_" + saveNum

#Define map settings
totBlocks = 60
mapWidth = 15
mapHeight = 15
waypointDist = 10
observationDist = 2 #Must be divisible by 2
obsPixleDensity = 10 #Number of pixles per unit cell of map at highest resolution
turnMemorySize = 6
allowedPositionError = 0.2
allowedBearingError = 15 #In degrees

showSuccessfull = True

stepReward = -1
spinReward = 0
collisionReward = -10
perfectWaypointPositionReward = 0
perfectWaypointBearingReward = 0
wayPointPositionReward = 10

plotEvery = 1
obsSize = (obsPixleDensity*observationDist*2)**2 + 2
allowedBearingError = allowedBearingError *np.pi/180
nActions = 2

maxSpeed = 0.1
maxTurnRate = np.pi/6

plotBestPath = False
plotObsRange = False
smoothInput = True
circleMap = True
allowClipping = False
useObservationSpace = True
requireBearing = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #WIll only be faster with lager observation space or batch size

print("Running program jobs on {}".format(device))

#Set up the display figure
fig = plt.figure()
axs = []
axs.append(plt.subplot2grid((3,5), (0, 0), colspan=3, rowspan=3))
axs.append(plt.subplot2grid((3,5), (0, 3), colspan=2, rowspan=2))
axs.append(plt.subplot2grid((3,5), (2, 3), colspan=2))
plt.ion()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0)

#Construct networks
#Actor
actorNetwork = Actor(obsSize,HIDDEN_SIZE, HIDDEN_SIZE2, nActions).to(device)
actorNetwork.load_state_dict(torch.load(savedActorPath))
actorNetwork.eval()

#Run training
episodeRewards = []
sucessfullEpisodes = 0
reachedOneWaypoint = 0
successfullMemories = []
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    #Reset environment
    episodeReward = 0.0
    previousPositions = []
    turnRateLog = [0, 0, 0, 0, 0, 0]
    waypointTarget = 0
    isDone = False
    mapObstacles, robot, goal, bestPath, waypoints, mapMatrix = resetMap(mapWidth, mapHeight, totBlocks, waypointDist, obsPixleDensity, circleMap)
    episodeIteration = 0
    episodeReachedWaypoint = False
    episodeMemory = []
    episodeRewardHistory = [0]

    #Generate the first state
    spaceMatrix = getObsSpaceRepresentation(mapMatrix, robot, (mapWidth, mapHeight), smoothInput, obsPixleDensity, observationDist)
    state = torch.DoubleTensor(spaceMatrix).unsqueeze(0).to(device)
    #additionalData = np.concatenate((np.array([waypoints[waypointTarget].position[0] - robot.position[0], waypoints[waypointTarget].position[1] - robot.position[1], waypoints[waypointTarget].yaw, robot.yaw, robot.getYawToPoint(waypoints[waypointTarget].position)]), np.array(turnRateLog)))
    additionalData = np.array([np.round(np.absolute(np.linalg.norm(waypoints[waypointTarget].position - robot.position)), 2), np.round(waypoints[waypointTarget].yaw - robot.yaw, 2), np.round(robot.getYawToPoint(waypoints[waypointTarget].position), 2)])
    additionalState = torch.DoubleTensor(additionalData).unsqueeze(0).to(device)

    while episodeIteration < MAX_EPISODE_ITERS:
        episodeIteration += 1

        #Get the next action
        action = actorNetwork(state, additionalState)
        action = action.detach().cpu().numpy()[0]
        """ 
        if random.random() < 0.1:
            #Do a random action
            u = random.uniform(0,1)
            w = random.uniform(-1,1)
            #update the action data for memory push
            action = np.array([u, w]) 
        else: """
        u = action[0]
        w = action[1]

        u = np.clip(u, 0., 1.)
        u = u*maxSpeed
        w = np.clip(w, -1., 1.)
        w = w*maxTurnRate

        turnRateLog.append(w)
        if len(turnRateLog) > turnMemorySize:
            turnRateLog = turnRateLog[1:turnMemorySize+1]

        #Compute the reward for the motion
        reward = robot.move(u, w, mapObstacles, mapWidth, mapHeight, stepReward, collisionReward, circleMap, allowClipping)

        #Check for waypoint reward
        reward = robot.getWaypointProximityReward(reward, waypoints[waypointTarget], wayPointPositionReward)

        #Check if the rolling average turn rate is too high (robot is spinning)
        if np.absolute(np.mean(turnRateLog)) > 0.9*maxTurnRate:
            #Robot is spinning
            reward += spinReward

        #Add some reward for pointing in the right direction
        reward -= np.absolute(robot.getYawToPoint(waypoints[waypointTarget].position))/(2*np.pi)*5
            
        episodeMemory.append((mapObstacles, mapWidth, mapHeight, copy.copy(robot.position), copy.copy(robot.yaw), robot.radius, goal, copy.deepcopy(previousPositions), bestPath, plotBestPath, copy.deepcopy(waypoints), observationDist, plotObsRange, copy.copy(waypointTarget), copy.copy(episodeIteration), MAX_EPISODE_ITERS, copy.copy(episodeRewardHistory)))

        #Check if waypoint is reached
        if robot.hasReachedWaypointPosition(waypoints[waypointTarget], allowedPositionError):
            reward += perfectWaypointPositionReward
            reward += 10*(0.1**((np.absolute(robot.getYawToPoint(waypoints[waypointTarget].position))/(2*np.pi))))
            if robot.hasReachedWaypointBearing(waypoints[waypointTarget], allowedBearingError) or not requireBearing:
                reward += perfectWaypointBearingReward
                waypoints[waypointTarget].reached = True
                waypointTarget += 1
                episodeIteration = 0
                if not episodeReachedWaypoint:
                    reachedOneWaypoint += 1
                    episodeReachedWaypoint = True 
                if waypointTarget == len(waypoints):
                    #Final goal has been reached
                    sucessfullEpisodes += 1
                    successfullMemories.append(episodeMemory)
                    isDone = True
                    waypointTarget -= 1 #Prevent fail on newAdditionalState creation
                    if episode%SEE_EPIDODE == 0:
                        print("\nWatched Success")


        #Get the new state with chosen action
        spaceMatrix = getObsSpaceRepresentation(mapMatrix, robot, (mapWidth, mapHeight), smoothInput, obsPixleDensity, observationDist)
        state = torch.DoubleTensor(spaceMatrix).unsqueeze(0).to(device)
        #newAdditionalData = np.concatenate((np.array([waypoints[waypointTarget].position[0] - robot.position[0], waypoints[waypointTarget].position[1] - robot.position[1], waypoints[waypointTarget].yaw, robot.yaw, robot.getYawToPoint(waypoints[waypointTarget].position)]), np.array(turnRateLog)))
        newAdditionalData = np.array([np.round(np.absolute(np.linalg.norm(waypoints[waypointTarget].position - robot.position)), 2), np.round(waypoints[waypointTarget].yaw - robot.yaw, 2), np.round(robot.getYawToPoint(waypoints[waypointTarget].position), 2)])
        additionalState = torch.DoubleTensor(newAdditionalData).unsqueeze(0).to(device)

        #Update the episode information
        episodeReward += reward
        episodeRewardHistory.append(episodeReward)



        if episode%SEE_EPIDODE == 0:
            #The last run of the batch is being computed

            if episodeIteration%10 == 0:
                #Save every 10 iterations to plot
                previousPositions.append(np.copy(robot.position))
            
            if episodeIteration%plotEvery == 0:
                #Plot enviromnent
                plotMap(axs[0], mapObstacles, mapWidth, mapHeight, robot, goal, previousPositions, bestPath, plotBestPath, waypoints, observationDist, plotObsRange, waypointTarget, circleMap)
                plotObservationInput(axs[1], spaceMatrix, obsPixleDensity, robot)
                plotLifespanBar(axs[2], mapWidth, episodeIteration, MAX_EPISODE_ITERS)
                plt.draw()
                plt.pause(0.0008)
                plt.show()
        
        if isDone:
            #Episode is complete
            episodeRewards.append(episodeReward)

            if episode%SEE_EPIDODE == 0:
                print("\nWatched reward: {}".format(episodeReward))
            break
    if not isDone:
        episodeRewards.append(episodeReward)
        if episode%SEE_EPIDODE == 0:
            print("\nWatched reward: {}".format(episodeReward))

print("###########################")
print("Model run complete")
print("###########################")
print("Episodes fully completed: {} of {}".format(sucessfullEpisodes, EPISODES))
print("Success rate: {}%".format(round(100*sucessfullEpisodes/EPISODES, 2)))
print("At least 1 waypoint reached: {} of {}".format(reachedOneWaypoint, EPISODES))
print("Success rate: {}%".format(round(100*reachedOneWaypoint/EPISODES, 2)))

if showSuccessfull:
    for successfullMemory in successfullMemories:
        for memoryFrame in successfullMemory:
            plotSuccess(axs[0], memoryFrame[0], memoryFrame[1], memoryFrame[2], memoryFrame[3], memoryFrame[4], memoryFrame[5], memoryFrame[6], memoryFrame[7], memoryFrame[8], memoryFrame[9], memoryFrame[10], memoryFrame[11], memoryFrame[12], memoryFrame[13], circleMap)
            plotSuccessReward(axs[1], memoryFrame[16])
            plotLifespanBar(axs[2], memoryFrame[1], memoryFrame[14], memoryFrame[15])
            plt.draw()
            plt.pause(0.1)
            plt.show()
