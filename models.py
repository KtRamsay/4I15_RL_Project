import numpy as np
import random

#Define a map block object
class MapBlock:
    def __init__(self, mapObstacles, mapWidth, mapHeight):
        self.width = 1
        self.height = 1

        freePosition = False
        while(freePosition==False):
            x1 = random.randint(0, mapWidth - self.width)
            y1 = random.randint(0, mapHeight - self.height)
            x2 = x1 + self.width
            y2 = y1 + self.height

            #Define block vertices, from bottom left anticlockwise

            self.vert1 = np.array([x1, y1])
            self.vert3 = np.array([x2, y2])

            freePosition = True
            for i in range(len(mapObstacles)):
                if self.checkBlockOverlap(mapObstacles[i]):
                    freePosition = False
                    break

            self.vert2 = np.array([x2, y1])
            self.vert4 = np.array([x1, y2])

    def checkBlockOverlap(self, block):
        if self.vert1[0] < block.vert3[0] and self.vert3[0] > block.vert1[0] and self.vert1[1] < block.vert3[1] and self.vert3[1] > block.vert1[1]:
            return True
        else:
            return False

#Define a map block object
class MapCircle:
    def __init__(self, mapObstacles, mapWidth, mapHeight):
        self.radius = round(random.uniform(0.1,0.5), 1)

        freePosition = False
        while(freePosition==False):
            x = random.uniform(self.radius, mapWidth - self.radius)
            y = random.uniform(self.radius, mapHeight - self.radius)

            self.position = np.array([x, y])

            freePosition = True
            for i in range(len(mapObstacles)):
                if self.checkCircleOverlap(mapObstacles[i]):
                    freePosition = False
                    break

    def checkCircleOverlap(self, circle):
        if np.absolute(np.linalg.norm(self.position - circle.position)) <= self.radius + circle.radius:
            return True
        else:
            return False

#Define a robot object
class Robot:
    def __init__(self, mapObstacles, mapWidth, mapHeight, circleMap):
        self.radius = 0.4

        valid = False
        self.yaw = random.uniform(0, 2*np.pi)
        self.setRotMatrix()
        attempts = 0
        while(valid==False):
            attempts += 1
            self.position = np.array([random.randint(0, mapWidth-2)+0.5, random.randint(0, mapHeight-2)+0.5], dtype="float")
            valid = True
            for i in range(len(mapObstacles)):
                if circleMap:
                    if self.checkCircleCollision(mapObstacles[i]):
                        valid = False
                else:
                    if self.checkBlockCollision(mapObstacles[i]):
                        valid = False
            if(attempts > 100):
                print("Failed to place robot/goal in map with 100 attempts, generating new map...")
                self.position = np.array([-1, -1])
                break

    def move(self, u, w, mapObstacles, mapWidth, mapHeight, stepReward, collisionReward, circleMap, allowClipping):
        reward = stepReward
        prevPos = np.copy(self.position)
        self.position[0] += u*np.cos(self.yaw)
        self.position[1] += u*np.sin(self.yaw)
        self.yaw = self.yaw + w
        self.setRotMatrix()

        bonk = False

        #Check for wall bonk
        if self.position[0] > mapWidth - self.radius or self.position[0] < self.radius:
            bonk = True
            #print("X Wall bonk")
        if self.position[1] > mapHeight - self.radius or self.position[1] < self.radius:
            bonk = True
            #print("Y Wall bonk")

        #Check for block bonk
        if not circleMap:
            for i in range(len(mapObstacles)):
                if self.checkBlockCollision(mapObstacles[i]):
                    bonk = True
                    #print("Block bonk")
                    break
        else:
            for i in range(len(mapObstacles)):
                if self.checkCircleCollision(mapObstacles[i]):
                    bonk = True
                    #print("Block bonk")
                    break
            

        if bonk==True and not allowClipping:
            self.position = prevPos
            reward = collisionReward

        return reward

    def setRotMatrix(self):
        #Rotation matrix for clockwise by yaw angle
        self.rotMatrix = np.array([[np.cos(self.yaw), np.sin(self.yaw)],[-np.sin(self.yaw), np.cos(self.yaw)]])


    def checkBlockCollision(self, block):
        if self.position[0] + self.radius > block.vert1[0] and self.position[1] + self.radius > block.vert1[1] and self.position[0] - self.radius < block.vert3[0] and self.position[1] - self.radius < block.vert3[1]:
            return True
        else:
            return False

    def checkCircleCollision(self, circle):
        if np.absolute(np.linalg.norm(self.position - circle.position)) <= self.radius + circle.radius:
            return True
        else:
            return False

    def getRelativePosition(self, position):
        return position - self.position

    def getRotatedRelativePosition(self, position):
        return np.matmul(self.rotMatrix, self.getRelativePosition(position))

    def __sub__(self, other):
        return self.position - other.position

    def hasReachedWaypointPosition(self, waypoint, allowedPositionError):
        if np.absolute(np.linalg.norm(self.position-waypoint.position)) < allowedPositionError:
            return True
        else:
            return False
            
    def hasReachedWaypointBearing(self, waypoint, allowedBearingError):
        if np.absolute(self.yaw - waypoint.yaw) < allowedBearingError:
            return True
        else:
            return False

    def getWaypointProximityReward(self, reward, waypoint, maxReward):
        #If the robot is overlapping the waypoint location then give it a proportion of the reward
        targetDist = np.absolute(np.linalg.norm(self.position-waypoint.position))
        return -targetDist
        if targetDist > self.radius*2:
            return reward
        else:
            return reward + 0.4**(targetDist/(self.radius*2))*maxReward


    def bearingErrorReward(self, waypoint, perfectWaypointBearingReward):
        return (0.5 - np.absolute(self.yaw - waypoint.yaw) / np.pi) * perfectWaypointBearingReward

    def getYawToPoint(self, point):
        #Get the yaw from horizontal right to the point from the robot position
        pointVec = point - self.position
        if pointVec[0] == 0:
            if pointVec[1] > 0:
                baseAngle = np.pi/2
            else:
                baseAngle = -np.pi/2
        else:
            baseAngle = np.arctan(np.absolute(pointVec[1])/np.absolute(pointVec[0]))
        if pointVec[0] >= 0:
            #To the right
            if pointVec[1] >= 0:
                #Upper right
                yawToPoint = baseAngle
            else:
                #Lower right 
                yawToPoint = 2*np.pi -baseAngle
        else:
            #To the left
            if pointVec[1] >= 0:
                #Upper left
                yawToPoint = np.pi - baseAngle
            else:
                #Lower left 
                yawToPoint = np.pi + baseAngle
        yawToPoint = yawToPoint % (2*np.pi)
        return yawToPoint - self.yaw


        
#Define a Node for the A* algorithm
class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return (self.position == other.position).all()

#Define a waypoint model
class Waypoint:
    def __init__(self, position, yaw, index):
        self.radius = 0.3
        self.position = position
        self.yaw = yaw
        self.index = index
        self.reached = False
