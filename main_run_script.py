from collections import namedtuple
import numpy as np
#from tensorboardX import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import math
import time
import os

from map_generation import resetMap
from observation_functions import getObsSpaceRepresentation
from plot_functions import plotLifespanBar, plotMap, plotObservationInput
from ornstein_uhlenbeck import ornstein_uhlenbeck

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter


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

# define critic network
class Critic(nn.Module):
    def __init__(self, obSize, numActions, hiddenSize, hiddenSize2):
        super(Critic, self).__init__()
        if useObservationSpace:
            self.conv1 = nn.Conv2d(1,5,5,1).double()
            self.conv2 = nn.Conv2d(5,18,3).double()
            self.conv3 = nn.Conv2d(18,3,3).double()

            self.flatten = nn.Flatten().double()

            #self.fc1 = nn.Linear(86 + numActions,80).double()
            self.fc1 = nn.Linear(78 + numActions,80).double()
        else:
            #self.fc1 = nn.Linear(11+numActions,80).double()
            self.fc1 = nn.Linear(3+numActions,80).double()
        self.fc2 = nn.Linear(80,80).double()
        self.fc3 = nn.Linear(80,1).double()


        
    def forward(self, spaceMatrix, additionalData, action):
        if useObservationSpace:
            spaceMatrix = spaceMatrix.view((spaceMatrix.shape[0], 1, spaceMatrix.shape[1], spaceMatrix.shape[1]))

            spaceMatrix = F.relu(self.conv1(spaceMatrix))
            spaceMatrix = F.avg_pool2d(spaceMatrix,2,2)
            spaceMatrix = F.relu(self.conv2(spaceMatrix))  
            spaceMatrix = F.avg_pool2d(spaceMatrix,2,2)
            spaceMatrix = F.relu(self.conv3(spaceMatrix))  
            spaceMatrix = F.avg_pool2d(spaceMatrix,2,1)
            observation = self.flatten(spaceMatrix)

            state = torch.cat((observation, additionalData, action), 1)
        else :
            state = torch.cat((additionalData, action), 1)
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        state = self.fc3(state)
        return state
    
#Create a replay memory class for training
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, waypointData, action, reward, terminal, newState, newWaypointData):
        newMemory = (state, waypointData, action, reward, terminal, newState, newWaypointData)
        #Add new memory
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = newMemory
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        #Get a random sample from the memory
        batch = random.sample(self.memory, batch_size)
        states, waypointData, actions, rewards, terminals, newStates, newWaypointData = map(np.stack, zip(*batch))

        return states, waypointData, actions, rewards, terminals, newStates, newWaypointData

    def __len__(self):
        return len(self.memory)

#Train the network
def optimizeModel():
    #If insufficent memory data to train, return
    if len(memory) < BATCH_SIZE:
        return

    #Get a sample of transitions from memory to train on
    stateBatch, waypointDataBatch, actionBatch, rewardBatch, terminalBatch, newStateBatch, newWaypointDataBatch = memory.sample(BATCH_SIZE)

    #Get the action values for each state
    stateBatch = torch.DoubleTensor(stateBatch).to(device)
    waypointDataBatch = torch.DoubleTensor(waypointDataBatch).to(device)
    actionBatch = torch.DoubleTensor(actionBatch).to(device)
    rewardBatch = torch.DoubleTensor(rewardBatch).unsqueeze(1).to(device)
    terminalBatch = torch.DoubleTensor(np.float32(terminalBatch)).unsqueeze(1).to(device)
    newStateBatch = torch.DoubleTensor(newStateBatch).to(device)
    newWaypointDataBatch = torch.DoubleTensor(newWaypointDataBatch).to(device)

    #CRITIC
    # Compute the next action values based on the new state
    newStateActionBatch = targetActorNetwork(newStateBatch, newWaypointDataBatch)
    targetQ = targetCriticNetwork(newStateBatch, newWaypointDataBatch, newStateActionBatch)
    y = (targetQ.detach()*GAMMA*(1.0-terminalBatch)) + rewardBatch
    q = criticNetwork(stateBatch, waypointDataBatch, actionBatch)

    qOptimizer.zero_grad()
    qLoss = MSE(q, y)
    qLoss.backward()
    qOptimizer.step()

    #ACTOR
    # Optimize the model
    policyOptimizer.zero_grad()
    policyLoss = -criticNetwork(stateBatch, waypointDataBatch, actorNetwork(stateBatch, waypointDataBatch))
    policyLoss = policyLoss.mean()
    policyLoss.backward()
    policyOptimizer.step()

    writer.add_scalar("policy_loss", policyLoss, episode*MAX_EPISODE_ITERS + episodeIteration)
    writer.add_scalar("q_loss", qLoss, episode*MAX_EPISODE_ITERS + episodeIteration)    

#Define learning model settings
HIDDEN_SIZE = 128
HIDDEN_SIZE2 = 100
BATCH_SIZE = 64
GAMMA = 0.5
TARGET_UPDATE = 10
MAX_EPISODE_ITERS = 80
EPISODES = 1000
SEE_EPIDODE = 200
EPSILON = 0.1#1
EPSILON_DECAY = 1/(MAX_EPISODE_ITERS*1000)
EPSILON_END = 0.1
SAVE_EVERY = 500

#Set if to load a model
setName = "ObservationSpaceTest"
savedModel = True
saveNum = "1619211529"

savedActorPath = "Model_Saves/" + setName + "/Actor_" + saveNum
savedCriticPath = "Model_Saves/" + setName + "/Critic_" + saveNum

#Define map settings
totBlocks = 4
mapWidth = 5
mapHeight = 5
waypointDist = 3
observationDist = 2 #Must be divisible by 2
obsPixleDensity = 10 #Number of pixles per unit cell of map at highest resolution
turnMemorySize = 6
allowedPositionError = 0.2
allowedBearingError = 15 #In degrees

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

plotBestPath = True
plotObsRange = False
smoothInput = True
circleMap = True
allowClipping = False
useObservationSpace = True
requireBearing = True

logPath = "logs/" + setName + "/"  +str(time.time())
writer = SummaryWriter(logPath, comment='-robot-pathfinder')
print("Writing logs to {}".format(logPath))
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

if savedModel:
    actorNetwork.load_state_dict(torch.load(savedActorPath))
    actorNetwork.eval()

targetActorNetwork = Actor(obsSize,HIDDEN_SIZE, HIDDEN_SIZE2, nActions).to(device)
targetActorNetwork.load_state_dict(actorNetwork.state_dict())
targetActorNetwork.eval()

#Critic
criticNetwork  = Critic(obsSize, nActions, HIDDEN_SIZE, HIDDEN_SIZE2).to(device)

if savedModel:
    criticNetwork.load_state_dict(torch.load(savedCriticPath))
    criticNetwork.eval()

targetCriticNetwork  = Critic(obsSize, nActions, HIDDEN_SIZE, HIDDEN_SIZE2).to(device)
targetCriticNetwork.load_state_dict(criticNetwork.state_dict())
targetCriticNetwork.eval()

qOptimizer  = optim.Adam(criticNetwork.parameters(),  lr=0.001)
policyOptimizer = optim.Adam(params = actorNetwork.parameters(),lr=0.001)
MSE = nn.MSELoss()

memory = ReplayMemory(10000)
episodeRewards = []
rollingSuccess = []

#Run training
totalSteps = 0
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    #Reset environment
    episodeReward = 0.0
    previousPositions = []
    turnRateLog = [0, 0, 0, 0, 0, 0]
    waypointTarget = 0
    isDone = False
    mapObstacles, robot, goal, bestPath, waypoints, mapMatrix = resetMap(mapWidth, mapHeight, totBlocks, waypointDist, obsPixleDensity, circleMap)
    episodeIteration = 0

    #Generate the first state
    spaceMatrix = getObsSpaceRepresentation(mapMatrix, robot, (mapWidth, mapHeight), smoothInput, obsPixleDensity, observationDist)
    state = torch.DoubleTensor(spaceMatrix).unsqueeze(0).to(device)
    #additionalData = np.concatenate((np.array([waypoints[waypointTarget].position[0] - robot.position[0], waypoints[waypointTarget].position[1] - robot.position[1], waypoints[waypointTarget].yaw, robot.yaw, robot.getYawToPoint(waypoints[waypointTarget].position)]), np.array(turnRateLog)))
    additionalData = np.array([np.round(np.absolute(np.linalg.norm(waypoints[waypointTarget].position - robot.position)), 2), np.round(waypoints[waypointTarget].yaw - robot.yaw, 2), np.round(robot.getYawToPoint(waypoints[waypointTarget].position), 2)])
    additionalState = torch.DoubleTensor(additionalData).unsqueeze(0).to(device)

    while episodeIteration < MAX_EPISODE_ITERS:
        episodeIteration += 1
        if EPSILON > EPSILON_END:
            EPSILON -= EPSILON_DECAY
        else:
            EPSILON = EPSILON_END
        

        #Get the next action
        action = actorNetwork(state, additionalState)
        action = action.detach().cpu().numpy()[0]

        if random.random() < EPSILON:
            #Do a random action
            u = random.uniform(0,1)
            w = random.uniform(-1,1)
            #update the action data for memory push
            action = np.array([u, w]) 
        else:
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

        #Check if waypoint is reached
        if robot.hasReachedWaypointPosition(waypoints[waypointTarget], allowedPositionError):
            reward += perfectWaypointPositionReward
            reward += 10*(0.1**((np.absolute(robot.getYawToPoint(waypoints[waypointTarget].position))/(2*np.pi))))
            if robot.hasReachedWaypointBearing(waypoints[waypointTarget], allowedBearingError) or not requireBearing:
                reward += perfectWaypointBearingReward
                waypoints[waypointTarget].reached = True
                waypointTarget += 1
                episodeIteration = 0
                if waypointTarget == len(waypoints):
                    #Final goal has been reached
                    print("\nGreat success")
                    isDone = True
                    waypointTarget -= 1 #Prevent fail on newAdditionalState creation
            else:
                reward += robot.bearingErrorReward(waypoints[waypointTarget], perfectWaypointBearingReward)


        #Get the new state with chosen action
        spaceMatrix = getObsSpaceRepresentation(mapMatrix, robot, (mapWidth, mapHeight), smoothInput, obsPixleDensity, observationDist)
        newState = torch.DoubleTensor(spaceMatrix).unsqueeze(0).to(device)
        #newAdditionalData = np.concatenate((np.array([waypoints[waypointTarget].position[0] - robot.position[0], waypoints[waypointTarget].position[1] - robot.position[1], waypoints[waypointTarget].yaw, robot.yaw, robot.getYawToPoint(waypoints[waypointTarget].position)]), np.array(turnRateLog)))
        newAdditionalData = np.array([np.round(np.absolute(np.linalg.norm(waypoints[waypointTarget].position - robot.position)), 2), np.round(waypoints[waypointTarget].yaw - robot.yaw, 2), np.round(robot.getYawToPoint(waypoints[waypointTarget].position), 2)])
        newAdditionalState = torch.DoubleTensor(newAdditionalData).unsqueeze(0).to(device)


        # Store the transition in memory
        memory.push(state.detach().cpu().numpy()[0], additionalState.detach().cpu().numpy()[0],
                    action, reward, isDone, newState.detach().cpu().numpy()[0],
                    newAdditionalState.detach().cpu().numpy()[0])

        #Move to the new state 
        state = newState
        additionalState = newAdditionalState

        #Update the episode information
        episodeReward += reward


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
                plt.pause(0.001)
                plt.show()
                
        
        #Train the model
        optimizeModel()
        if isDone:
            #Episode is complete
            episodeRewards.append(episodeReward)
            rollingSuccess.append(1)
            writer.add_scalar("episode_reward", episodeReward, episode)

            if episode%SEE_EPIDODE == 0:
                print("\nWatched reward: {}".format(episodeReward))
            break
    if not isDone:
        episodeRewards.append(episodeReward)
        rollingSuccess.append(0)
        writer.add_scalar("episode_reward", episodeReward, episode)
        if episode%SEE_EPIDODE == 0:
            print("\nWatched reward: {}".format(episodeReward))

    
    if len(rollingSuccess) > 50:
        rollingSuccess = rollingSuccess[1:51]
        
    if len(rollingSuccess) == 50:
        averageSuccess = np.mean(rollingSuccess)*100
    else :
        averageSuccess = 0
    writer.add_scalar("rolling_average_success", averageSuccess, episode)

    # Update the target network from the training network
    if episode % TARGET_UPDATE == 0:
        targetActorNetwork.load_state_dict(actorNetwork.state_dict())
        targetCriticNetwork.load_state_dict(criticNetwork.state_dict())
        writer.flush()

    # Save the model
    if episode % SAVE_EVERY == 0:
        timeStamp = time.time()
        filePath = "Model_Saves/" + setName + "/"
        directory = os.path.dirname(filePath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        actorPath = filePath + "Actor_" + str(round(timeStamp))
        criticPath = filePath + "Critic_" + str(round(timeStamp))
        torch.save(actorNetwork.state_dict(), actorPath)
        torch.save(criticNetwork.state_dict(), criticPath)

writer.close()
