from models import MapBlock, MapCircle, Robot, Waypoint
from waypoint_generation import getBestPath, genWaypoints

import numpy as np
from PIL import Image, ImageDraw

def resetMap(mapWidth, mapHeight, totBlocks, waypointDist, obsPixleDensity, circleMap):
    printAll = False
    #Generate the map 
    if printAll:
        print("Generating map...")
    validMap = False
    while(validMap==False):
        validMap = True

        #Populate the map with blocks
        mapObstacles = []
        for i in range(totBlocks):
            if circleMap:
                mapObstacles.append(MapCircle(mapObstacles, mapWidth, mapHeight))
            else:
                mapObstacles.append(MapBlock(mapObstacles, mapWidth, mapHeight))
        else:
            if printAll:
                print("[1]:Map blocks placed")
            
        #Create the robot start location
        robot = Robot(mapObstacles, mapWidth, mapHeight, circleMap)
        if((robot.position == np.array([-1, -1])).all()):
            #Robot could not be placed, generate a new map
            validMap = False
            continue
        else:
            if printAll:
                print("[2]:Robot placed")

        #Create the goal
        goalBot = Robot(mapObstacles, mapWidth, mapHeight, circleMap)
        if((goalBot.position == np.array([-1, -1])).all() or (goalBot.position == robot.position).all()):
            #Goal could not be placed, generate a new map
            validMap = False
            continue
        else:
            if printAll:
                print("[3]:Goal placed")
            goal = Waypoint(goalBot.position, goalBot.yaw, 0)

        #Create the best path from A*
        bestPath = getBestPath(mapObstacles, mapWidth, mapHeight, robot, goal, circleMap)
        if(len(bestPath)==0):
            #No path could be found
            if printAll:
                print("No possible path from start to goal, generating a new map..")
            validMap = False
            continue
        else:
            if printAll:
                print("[4]:Best path found")

    #Split the best path into waypoint goals
    waypoints = genWaypoints(bestPath, waypointDist, goal)
    
    #
    mapMatrix = constructMap(mapWidth, mapHeight, mapObstacles, obsPixleDensity, circleMap)
    if printAll:
        print("Map generation completed")

    return mapObstacles, robot, goal, bestPath, waypoints, mapMatrix

# constructs a representation of the map
def constructMap(mapWidth, mapHeight, obstacles, obsPixleDensity, circleMap):
    # initialises blank map (i.e. all zeros)
    env = np.ones((mapWidth*50, mapHeight*50), dtype= "uint")

    # identifies any pixel that is covered by an obstacle
    if not circleMap:
        for block in obstacles:
            for x in range(block.vert1[0]*50, block.vert3[0]*50):
                for y in range(block.vert1[1]*50, block.vert3[1]*50):
                    env[x,y] = 0

    # convert array to image
    env = Image.frombytes(mode='1', size=env.shape[::-1], data=np.packbits(env, axis=1))
    if circleMap:
        draw = ImageDraw.Draw(env)
        for circle in obstacles:
            draw.ellipse([round(np.floor(circle.position[1]*50 - circle.radius*50)), round(np.floor(circle.position[0]*50 - circle.radius*50)), round(np.ceil(circle.position[1]*50 + circle.radius*50)), round(np.ceil(circle.position[0]*50 + circle.radius*50))], fill=0)
    env = env.rotate(90)
    #env.show()
    # resize
    env = env.resize((mapWidth*obsPixleDensity, mapHeight*obsPixleDensity))
    return env
