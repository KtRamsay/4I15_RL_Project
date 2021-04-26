import numpy as np

from models import Node, Waypoint

#Generate a matrix map from the map information
def genEnv(mapObstacles, mapWidth, mapHeight, circleMap):
    env = np.zeros((mapWidth,mapHeight))

    if circleMap:
        for i in range(len(mapObstacles)):
            startX = round(np.floor(mapObstacles[i].position[0] - mapObstacles[i].radius))
            endX = round(np.ceil(mapObstacles[i].position[0] + mapObstacles[i].radius))
            startY = round(np.floor(mapObstacles[i].position[1] - mapObstacles[i].radius))
            endY = round(np.ceil(mapObstacles[i].position[1] + mapObstacles[i].radius))
            
            for xDist in range(endX - startX):
                for yDist in range(endY - startY):
                    env[startX + xDist][startY + yDist] = 1

    else:
        for i in range(len(mapObstacles)):
            block = mapObstacles[i]
            for xDist in range(block.width):
                for yDist in range(block.height):
                    env[block.vert1[0] + xDist][block.vert1[1] + yDist] = 1

    return env

#Centre the path points to the middle of their squares
def cleanPath(path, robot, goal):
    newPath = []
    for i in range(len(path)):
        nextIndex = len(path)-1 - i
        newPath.append(np.array([path[nextIndex][0]+0.5, path[nextIndex][1]+0.5]))
    return newPath

#Generate waypoints 
def genWaypoints(path, waypointDist, goal):
    waypoints = []
    for i in range(len(path)):
        if i > 0 and i%waypointDist==0 and i<len(path)-1:
            xToAvg = []
            yToAvg = []
            for nextPointIndex in range(3):
                if i + 1 + nextPointIndex < len(path):
                    xToAvg.append(path[i+1+nextPointIndex][0])
                    yToAvg.append(path[i+1+nextPointIndex][1])
            avgX = np.average(xToAvg)
            avgY = np.average(yToAvg)

            xVec = avgX-path[i][0]
            yVec = avgY-path[i][1]

            if xVec ==0:
                #Directly above or bellow
                if yVec > 0:
                    #Above  
                    waypointYaw = np.pi/2
                else:
                    #Bellow
                    waypointYaw = 1.5*np.pi
            else:
                baseYaw = np.arctan(np.absolute(yVec)/np.absolute(xVec))

                if xVec > 0:
                    #To the right
                    if yVec > 0:
                        #Upper right  
                        waypointYaw = baseYaw
                    else:
                        #Lower right
                        waypointYaw = 2*np.pi - baseYaw
                elif xVec <0:
                    #To the left
                    if yVec > 0:
                        #Upper left  
                        waypointYaw = np.pi - baseYaw
                    else:
                        #Lower left
                        waypointYaw = np.pi + baseYaw

            waypoints.append(Waypoint((path[i][0], path[i][1]), waypointYaw, len(waypoints)))

    goal.index = len(waypoints)
    waypoints.append(goal)
    return waypoints

#Use A* to generate a path to the goal
def getBestPath(mapObstacles, mapWidth, mapHeight, robot, goal, circleMap):
    #Generate the map
    env = genEnv(mapObstacles, mapWidth, mapHeight, circleMap)

    startPos = np.array([robot.position[0]-0.5, robot.position[1]-0.5])
    goalPos = np.array([goal.position[0]-0.5, goal.position[1]-0.5])

    #Create start and goal nodes
    startNode = Node(None, startPos)
    startNode.g = startNode.h = startNode.f = 0
    endNode = Node(None, goalPos)
    endNode.g = endNode.h = endNode.f = 0

    openList = []
    closedList = []

    openList.append(startNode)

    while len(openList) > 0:
        #Find best open node
        currentNodeIndex = 0
        for i in range(len(openList)):
            if openList[i].f < openList[currentNodeIndex].f:
                currentNodeIndex = i
        currentNode = openList[currentNodeIndex]

        openList.pop(currentNodeIndex)
        closedList.append(currentNode)

        #Check if the goal is reached
        if currentNode == endNode:
            path = []
            current = currentNode
            while current is not None:
                path.append(current.position)
                current = current.parent
            return  cleanPath(path, robot, goal)

        #Get node children
        children = []
        for newPos in [np.array([0, -1]), np.array([0, 1]), np.array([-1, 0]),np.array([1, 0])]:
            nodePos = np.array([round(currentNode.position[0]+newPos[0]), round(currentNode.position[1]+newPos[1])])
            #Check node within boundary
            if nodePos[0] > (mapWidth-1) or nodePos[0] < 0 or nodePos[1] > mapHeight-1 or nodePos[1] < 0:
                continue

            #Check if obstructed
            if env[nodePos[0]][nodePos[1]] != 0:
                continue

            # Create new node
            newNode = Node(currentNode, nodePos)

            # Append
            children.append(newNode)

        for child in children:
            
            #Check if child is already closed
            if len([closedChild for closedChild in closedList if closedChild == child]) > 0:
                continue

            #Gen f, g and h
            child.g = currentNode.g + 1
            child.h = ((child.position[0] - endNode.position[0]) ** 2) + ((child.position[1] - endNode.position[1]) ** 2)
            child.f = child.g + child.h

            #Check if child is already open with a lower value
            if len([openChild for openChild in openList if openChild == child and child.g > openChild.g]) > 0:
                continue

            #Otherwise open the child
            openList.append(child)
    return []
