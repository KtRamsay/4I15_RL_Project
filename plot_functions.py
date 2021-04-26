import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#Define the map plotting function
def plotMap(axes, mapObstacles, mapWidth, mapHeight, robot, goal, previousPositions, bestPath, plotBestPath, waypoints, observationDist, plotObsRange, waypointTarget, circleMap):
    axes.clear()

    #Plot the map border
    mapLimitsOuter = plt.Polygon([(-1, -1), (mapWidth+1, -1), (mapWidth+1, mapHeight+1), (-1, mapHeight+1)], closed=True, fc='black', ec='black')
    axes.add_patch(mapLimitsOuter)
    mapLimitsInner = plt.Polygon([(0, -0), (mapWidth, 0), (mapWidth, mapHeight), (0, mapHeight)], closed=True, fc='white', ec='black')
    axes.add_patch(mapLimitsInner)

    #Plot the map blocks
    if circleMap:
        for i in range(len(mapObstacles)):
            circle = plt.Circle(mapObstacles[i].position, mapObstacles[i].radius, fc='black', ec='black')
            axes.add_patch(circle)
    else:
        for i in range(len(mapObstacles)):
            block = plt.Polygon([mapObstacles[i].vert1, mapObstacles[i].vert2, mapObstacles[i].vert3, mapObstacles[i].vert4], closed=True, fc='black', ec='black')
            axes.add_patch(block)

    if plotBestPath:
        #Plot the best path
        for i in range(len(bestPath)):
            if i > 0:
                positonLine = plt.Polygon([bestPath[i],bestPath[i-1]], closed=True, fill=None, ec='limegreen')
                axes.add_patch(positonLine)
            positionCircle = plt.Circle(bestPath[i], 0.05, fc='darkgreen', ec='limegreen')
            axes.add_patch(positionCircle)

    #Plot the waypoints
    for i in range(len(waypoints)):
        point = waypoints[i]
        if point.reached:
            waypointBody = plt.Circle((point.position), point.radius, fc='gold', ec='yellow')
            waypointDirection = plt.Polygon([point.position,(point.position[0] + point.radius*np.cos(point.yaw), point.position[1] + point.radius*np.sin(point.yaw))], closed=True, fill=None, ec='yellow')
        else:
            waypointBody = plt.Circle((point.position), point.radius, fc='slateblue', ec='lavender')
            waypointDirection = plt.Polygon([point.position,(point.position[0] + point.radius*np.cos(point.yaw), point.position[1] + point.radius*np.sin(point.yaw))], closed=True, fill=None, ec='lavender')
        axes.add_patch(waypointBody)
        axes.add_patch(waypointDirection)

    #Plot the goal location
    goalCircle = plt.Circle(goal.position, goal.radius, fc='pink', ec='magenta')
    axes.add_patch(goalCircle)
    goalLine = plt.Polygon([goal.position,(goal.position[0] + goal.radius*np.cos(goal.yaw), goal.position[1] + goal.radius*np.sin(goal.yaw))], closed=True, fill=None, ec='green')
    axes.add_patch(goalLine)

    #Plot the previous path
    for i in range(len(previousPositions)):
        if i > 0:
            positonLine = plt.Polygon([previousPositions[i],previousPositions[i-1]], closed=True, fill=None, ec='violet')
            axes.add_patch(positonLine)
        positionCircle = plt.Circle(previousPositions[i], 0.1, fc='violet', ec='blueviolet')
        axes.add_patch(positionCircle)

    #Plot the robot location
    robotCircle = plt.Circle(robot.position, robot.radius, fc='green', ec='blue')
    axes.add_patch(robotCircle)
    robotLine = plt.Polygon([robot.position,(robot.position[0] + robot.radius*np.cos(robot.yaw), robot.position[1] + robot.radius*np.sin(robot.yaw))], closed=True, fill=None, ec='magenta')
    axes.add_patch(robotLine)

    #Highloght the current target waypoint
    targetCircle = plt.Circle(waypoints[waypointTarget].position, 0.9, fc='None', ec='red')
    axes.add_patch(targetCircle)

    if (plotObsRange):
        #Plot the observation area
        anticlockRotMat = np.array([[np.cos(robot.yaw), -np.sin(robot.yaw)],[np.sin(robot.yaw), np.cos(robot.yaw)]])
        obsArea = plt.Polygon([robot.position + np.matmul(anticlockRotMat, np.array([-observationDist, -observationDist])), robot.position + np.matmul(anticlockRotMat, np.array([-observationDist, observationDist])), robot.position + np.matmul(anticlockRotMat, np.array([observationDist, observationDist])), robot.position + np.matmul(anticlockRotMat, np.array([observationDist, -observationDist]))], closed=True, fill=None, ec='red')
        axes.add_patch(obsArea)

    axes.title.set_text("Robot map")
    if (plotObsRange):
        axes.set_xlim(-np.sqrt(2)*observationDist, np.sqrt(2)*observationDist + mapWidth)
        axes.set_ylim(-np.sqrt(2)*observationDist, np.sqrt(2)*observationDist + mapHeight)
    else:
        axes.axis('scaled')
    axes.axis('off')

#Define a lifespan bar plot
def plotLifespanBar(axes, mapWidth, iteration, maxIteration):
    axes.clear()
    emptyLifespanBar = plt.Rectangle((-0.75, -0.75), mapWidth+1.5, 0.5, fc='grey', ec='grey')
    axes.add_patch(emptyLifespanBar)
    fillingLifespanBar = plt.Rectangle((-0.75, -0.75), (mapWidth+1.5)*iteration/maxIteration, 0.5, fc='limegreen', ec='grey')
    axes.add_patch(fillingLifespanBar)

    axes.title.set_text("Robot Lifespan")
    axes.axis('scaled')
    axes.axis('off')

def plotObservationInput(axes, spaceMatrix, obsPixleDensity, robot):
    axes.clear()
    
    #Create image
    obsImg = Image.fromarray(spaceMatrix*255)
    axes.imshow(obsImg)

    #Plot the robot location
    robotCircle = plt.Circle((len(spaceMatrix)/2, len(spaceMatrix)/2), robot.radius*obsPixleDensity, fill=None, ec='red')
    axes.add_patch(robotCircle)
    robotLine = plt.Polygon([(len(spaceMatrix)/2, len(spaceMatrix)/2), (len(spaceMatrix)/2, len(spaceMatrix)/2-robot.radius*obsPixleDensity)], closed=True, fill=None, ec='red')
    axes.add_patch(robotLine)

    """ for i in range(10):
        rect = plt.Polygon([(i*4-0.5, -0.5), (i*4-0.5, 3.5), (i*4+3.5, 3.5), (i*4+3.5, -0.5)], closed=True, fill=None, ec='red')
        axes.add_patch(rect)
        rect = plt.Polygon([(i*4-0.5, 35.5), (i*4-0.5, 39.5), (i*4+3.5, 39.5), (i*4+3.5, 35.5)], closed=True, fill=None, ec='red')
        axes.add_patch(rect)
        rect = plt.Polygon([(-0.5, i*4-0.5), (3.5, i*4-0.5), (3.5, i*4+3.5), (-0.5, i*4+3.5)], closed=True, fill=None, ec='red')
        axes.add_patch(rect)
        rect = plt.Polygon([(35.5, i*4-0.5), (39.5, i*4-0.5), (39.5, i*4+3.5), (35.5, i*4+3.5)], closed=True, fill=None, ec='red')
        axes.add_patch(rect)
        
    for j in range(15):
        i = j+2
        rect = plt.Polygon([(i*2-0.5, 3.5), (i*2-0.5, 5.5), (i*2+3.5, 5.5), (i*2+3.5, 3.5)], closed=True, fill=None, ec='red')
        axes.add_patch(rect)
        rect = plt.Polygon([(i*2-0.5, 33.5), (i*2-0.5, 35.5), (i*2+3.5, 35.5), (i*2+3.5, 33.5)], closed=True, fill=None, ec='red')
        axes.add_patch(rect)
        rect = plt.Polygon([(3.5, i*2-0.5), (5.5, i*2-0.5), (5.5, i*2+3.5), (3.5, i*2+3.5)], closed=True, fill=None, ec='red')
        axes.add_patch(rect)
        rect = plt.Polygon([(33.5, i*2-0.5), (35.5, i*2-0.5), (35.5, i*2+3.5), (33.5, i*2+3.5)], closed=True, fill=None, ec='red')
        axes.add_patch(rect)
        
        rect = plt.Polygon([(i*2-0.5, 5.5), (i*2-0.5, 7.5), (i*2+3.5, 7.5), (i*2+3.5, 5.5)], closed=True, fill=None, ec='red')
        axes.add_patch(rect)
        rect = plt.Polygon([(i*2-0.5, 31.5), (i*2-0.5, 33.5), (i*2+3.5, 33.5), (i*2+3.5, 31.5)], closed=True, fill=None, ec='red')
        axes.add_patch(rect)
        rect = plt.Polygon([(5.5, i*2-0.5), (7.5, i*2-0.5), (7.5, i*2+3.5), (5.5, i*2+3.5)], closed=True, fill=None, ec='red')
        axes.add_patch(rect)
        rect = plt.Polygon([(31.5, i*2-0.5), (33.5, i*2-0.5), (33.5, i*2+3.5), (31.5, i*2+3.5)], closed=True, fill=None, ec='red')
        axes.add_patch(rect)
    for i in range(24):
        for j in range(24):
            rect = plt.Polygon([(i+7.5, j+7.5), (i+7.5, j+8.5), (i+8.5, j+8.5), (i+8.5, j+7.5)], closed=True, fill=None, ec='red')
            axes.add_patch(rect)
 """
    axes.title.set_text("Robot observation input")
    axes.axis('scaled')
    axes.axis('off')

#Define the success plotting function
def plotSuccess(axes, mapObstacles, mapWidth, mapHeight, robotPosition, robotYaw, robotRadius, goal, previousPositions, bestPath, plotBestPath, waypoints, observationDist, plotObsRange, waypointTarget, circleMap):
    axes.clear()

    #Plot the map border
    mapLimitsOuter = plt.Polygon([(-1, -1), (mapWidth+1, -1), (mapWidth+1, mapHeight+1), (-1, mapHeight+1)], closed=True, fc='black', ec='black')
    axes.add_patch(mapLimitsOuter)
    mapLimitsInner = plt.Polygon([(0, -0), (mapWidth, 0), (mapWidth, mapHeight), (0, mapHeight)], closed=True, fc='white', ec='black')
    axes.add_patch(mapLimitsInner)

    #Plot the map blocks
    if circleMap:
        for i in range(len(mapObstacles)):
            circle = plt.Circle(mapObstacles[i].position, mapObstacles[i].radius, fc='black', ec='black')
            axes.add_patch(circle)
    else:
        for i in range(len(mapObstacles)):
            block = plt.Polygon([mapObstacles[i].vert1, mapObstacles[i].vert2, mapObstacles[i].vert3, mapObstacles[i].vert4], closed=True, fc='black', ec='black')
            axes.add_patch(block)

    if plotBestPath:
        #Plot the best path
        for i in range(len(bestPath)):
            if i > 0:
                positonLine = plt.Polygon([bestPath[i],bestPath[i-1]], closed=True, fill=None, ec='limegreen')
                axes.add_patch(positonLine)
            positionCircle = plt.Circle(bestPath[i], 0.05, fc='darkgreen', ec='limegreen')
            axes.add_patch(positionCircle)

    #Plot the waypoints
    for i in range(len(waypoints)):
        point = waypoints[i]
        if point.reached:
            waypointBody = plt.Circle((point.position), point.radius, fc='gold', ec='yellow')
            waypointDirection = plt.Polygon([point.position,(point.position[0] + point.radius*np.cos(point.yaw), point.position[1] + point.radius*np.sin(point.yaw))], closed=True, fill=None, ec='yellow')
        else:
            waypointBody = plt.Circle((point.position), point.radius, fc='slateblue', ec='lavender')
            waypointDirection = plt.Polygon([point.position,(point.position[0] + point.radius*np.cos(point.yaw), point.position[1] + point.radius*np.sin(point.yaw))], closed=True, fill=None, ec='lavender')
        axes.add_patch(waypointBody)
        axes.add_patch(waypointDirection)

    #Plot the goal location
    goalCircle = plt.Circle(goal.position, goal.radius, fc='pink', ec='magenta')
    axes.add_patch(goalCircle)
    goalLine = plt.Polygon([goal.position,(goal.position[0] + goal.radius*np.cos(goal.yaw), goal.position[1] + goal.radius*np.sin(goal.yaw))], closed=True, fill=None, ec='green')
    axes.add_patch(goalLine)

    #Plot the previous path
    for i in range(len(previousPositions)):
        if i > 0:
            positonLine = plt.Polygon([previousPositions[i],previousPositions[i-1]], closed=True, fill=None, ec='violet')
            axes.add_patch(positonLine)
        positionCircle = plt.Circle(previousPositions[i], 0.1, fc='violet', ec='blueviolet')
        axes.add_patch(positionCircle)

    #Plot the robot location
    robotCircle = plt.Circle(robotPosition, robotRadius, fc='green', ec='blue')
    axes.add_patch(robotCircle)
    robotLine = plt.Polygon([robotPosition,(robotPosition[0] + robotRadius*np.cos(robotYaw), robotPosition[1] + robotRadius*np.sin(robotYaw))], closed=True, fill=None, ec='magenta')
    axes.add_patch(robotLine)

    #Highloght the current target waypoint
    targetCircle = plt.Circle(waypoints[waypointTarget].position, 0.9, fc='None', ec='red')
    axes.add_patch(targetCircle)

    if (plotObsRange):
        #Plot the observation area
        anticlockRotMat = np.array([[np.cos(robot.yaw), -np.sin(robot.yaw)],[np.sin(robot.yaw), np.cos(robot.yaw)]])
        obsArea = plt.Polygon([robot.position + np.matmul(anticlockRotMat, np.array([-observationDist, -observationDist])), robot.position + np.matmul(anticlockRotMat, np.array([-observationDist, observationDist])), robot.position + np.matmul(anticlockRotMat, np.array([observationDist, observationDist])), robot.position + np.matmul(anticlockRotMat, np.array([observationDist, -observationDist]))], closed=True, fill=None, ec='red')
        axes.add_patch(obsArea)

    axes.title.set_text("Robot map")
    if (plotObsRange):
        axes.set_xlim(-np.sqrt(2)*observationDist, np.sqrt(2)*observationDist + mapWidth)
        axes.set_ylim(-np.sqrt(2)*observationDist, np.sqrt(2)*observationDist + mapHeight)
    else:
        axes.axis('scaled')
    axes.axis('off')
    
def plotSuccessReward(axes, episodeRewardHistory):
    axes.clear()
    axes.title.set_text("Episode Reward")
    axes.set_xlabel('episode step')
    axes.set_ylabel('total reward')
    axes.plot(episodeRewardHistory)
    axes.axis('scaled')
