import numpy as np

#Get image matrix of observation space
def getObsSpaceRepresentation(env, robot, mapSize, smoothInput, obsPixleDensity, observationDist):
    robotPosition = robot.position
    robotOrientation = robot.yaw

    robotPosition = robotPosition * (np.divide((mapSize[0]*obsPixleDensity, mapSize[1]*obsPixleDensity), mapSize))
    xR, yR = (robotPosition[0], mapSize[1]*obsPixleDensity - robotPosition[1])
    env = env.crop((xR - mapSize[0]*obsPixleDensity, yR - mapSize[1]*obsPixleDensity, xR + mapSize[0]*obsPixleDensity, yR + mapSize[1]*obsPixleDensity))

    env = env.rotate(- (np.rad2deg(robotOrientation) - 90))

    w, h = np.divide(env.size, (2, 2))
    cropSize = obsPixleDensity*observationDist
    env = env.crop((w - cropSize, h - cropSize, w + cropSize, h + cropSize))
    mapMatrix = (np.array(env))
    mapMatrix = mapMatrix.astype(int)

    if smoothInput:
        return smoothSpaceMatrix(np.asarray(mapMatrix, dtype="float64"))
    else:
        return np.asarray(mapMatrix, dtype="float64")

#Aveage over a section of the given matrix
def smoothRegion(mat, binStartVer, binEndVer, binStartHor, binEndHor):
    mat[round(binStartVer):round(binEndVer), round(binStartHor):round(binEndHor)] = round(mat[round(binStartVer):round(binEndVer), round(binStartHor):round(binEndHor)].mean())
    return mat

#Average further pixel groups together to reduce amount of information
def smoothSpaceMatrix(spaceMatrix):
    smallNum = 40 # Must be divisible by 4

    medNum = round(smallNum/2)
    largeNum = round(medNum/2)

    smoothedMatrix = np.copy(spaceMatrix)
    largeSpacing = np.floor(len(spaceMatrix)/largeNum)
    medSpacing = np.floor(len(spaceMatrix)/medNum)
    smallSpacing = np.floor(len(spaceMatrix)/smallNum)

    #Deal with large sqaures
    for i in range(largeNum-1):
        #Top Horizaontal squares
        smoothedMatrix = smoothRegion(smoothedMatrix, 0, largeSpacing, i*largeSpacing, (i+1)*largeSpacing)
        #Bottom Horizaontal squares
        smoothedMatrix = smoothRegion(smoothedMatrix, (largeNum-1)*largeSpacing, len(smoothedMatrix), i*largeSpacing, (i+1)*largeSpacing)
        #Left Vertical squares
        smoothedMatrix = smoothRegion(smoothedMatrix, i*largeSpacing, (i+1)*largeSpacing, 0, largeSpacing)
        #Right Vertical squares
        smoothedMatrix = smoothRegion(smoothedMatrix, i*largeSpacing, (i+1)*largeSpacing, (largeNum-1)*largeSpacing, len(smoothedMatrix))
    #Bottom right large
    smoothedMatrix = smoothRegion(smoothedMatrix, (largeNum-1)*largeSpacing, len(smoothedMatrix), (largeNum-1)*largeSpacing, len(smoothedMatrix))
    
    #Deal with medium squares
    for i in range(medNum-5):
        #Top Horizaontal squares
        smoothedMatrix = smoothRegion(smoothedMatrix, largeSpacing, largeSpacing + medSpacing, largeSpacing + i*medSpacing, largeSpacing + (i+1)*medSpacing)
        smoothedMatrix = smoothRegion(smoothedMatrix, largeSpacing + medSpacing, largeSpacing + 2*medSpacing, largeSpacing + i*medSpacing, largeSpacing + (i+1)*medSpacing)
        #Bottom Horizaontal squares
        smoothedMatrix = smoothRegion(smoothedMatrix, largeSpacing + (medNum-6)*medSpacing, largeSpacing + (medNum-5)*medSpacing, largeSpacing + i*medSpacing, largeSpacing + (i+1)*medSpacing)
        smoothedMatrix = smoothRegion(smoothedMatrix, largeSpacing + (medNum-5)*medSpacing, (largeNum-1)*largeSpacing, largeSpacing + i*medSpacing, largeSpacing + (i+1)*medSpacing)
        #Left Vertical squares
        smoothedMatrix = smoothRegion(smoothedMatrix, largeSpacing + i*medSpacing, largeSpacing + (i+1)*medSpacing, largeSpacing, largeSpacing + medSpacing)
        smoothedMatrix = smoothRegion(smoothedMatrix, largeSpacing + i*medSpacing, largeSpacing + (i+1)*medSpacing, largeSpacing + medSpacing, largeSpacing + 2*medSpacing)
        #Right Vertical squares
        smoothedMatrix = smoothRegion(smoothedMatrix, largeSpacing + i*medSpacing, largeSpacing + (i+1)*medSpacing, largeSpacing + (medNum-6)*medSpacing, largeSpacing + (medNum-5)*medSpacing)
        smoothedMatrix = smoothRegion(smoothedMatrix, largeSpacing + i*medSpacing, largeSpacing + (i+1)*medSpacing, largeSpacing + (medNum-5)*medSpacing, (largeNum-1)*largeSpacing)
    #Bottom right square
    smoothedMatrix = smoothRegion(smoothedMatrix, largeSpacing + (medNum-5)*medSpacing, (largeNum-1)*largeSpacing, largeSpacing + (medNum-5)* medSpacing, (largeNum-1)*largeSpacing)
    
    #Deal with small squares
    for i in range(smallNum - 17):
        #Top Horizaontal squares
        smoothedMatrix = smoothRegion(smoothedMatrix, largeSpacing+2*medSpacing, largeSpacing+2*medSpacing + smallSpacing, largeSpacing+2*medSpacing + i*smallSpacing, largeSpacing+2*medSpacing + (i+1)*smallSpacing)
        smoothedMatrix = smoothRegion(smoothedMatrix, largeSpacing+2*medSpacing+smallSpacing, largeSpacing+2*medSpacing + 2*smallSpacing, largeSpacing+2*medSpacing + i*smallSpacing, largeSpacing+2*medSpacing + (i+1)*smallSpacing)
        #Bottom Horizaontal squares
        smoothedMatrix = smoothRegion(smoothedMatrix, largeSpacing + 2*medSpacing + (smallNum - 18)*smallSpacing, largeSpacing + 2*medSpacing + (smallNum - 17)*smallSpacing, largeSpacing+2*medSpacing + i*smallSpacing, largeSpacing+2*medSpacing + (i+1)*smallSpacing)
        smoothedMatrix = smoothRegion(smoothedMatrix, largeSpacing + 2*medSpacing + (smallNum - 17)*smallSpacing, (largeNum-2)*largeSpacing, largeSpacing+2*medSpacing + i*smallSpacing, largeSpacing+2*medSpacing + (i+1)*smallSpacing)
        #Left Vertical squares
        smoothedMatrix = smoothRegion(smoothedMatrix, largeSpacing+2*medSpacing + i*smallSpacing, largeSpacing+2*medSpacing + (i+1)*smallSpacing, largeSpacing+2*medSpacing, largeSpacing+2*medSpacing + smallSpacing)
        smoothedMatrix = smoothRegion(smoothedMatrix, largeSpacing+2*medSpacing + i*smallSpacing, largeSpacing+2*medSpacing + (i+1)*smallSpacing, largeSpacing+2*medSpacing+smallSpacing, largeSpacing+2*medSpacing + 2*smallSpacing)
        #Right Vertical squares
        smoothedMatrix = smoothRegion(smoothedMatrix, largeSpacing+2*medSpacing + i*smallSpacing, largeSpacing+2*medSpacing + (i+1)*smallSpacing, largeSpacing + 2*medSpacing + (smallNum - 18)*smallSpacing, largeSpacing + 2*medSpacing + (smallNum - 17)*smallSpacing)
        smoothedMatrix = smoothRegion(smoothedMatrix, largeSpacing+2*medSpacing + i*smallSpacing, largeSpacing+2*medSpacing + (i+1)*smallSpacing, largeSpacing + 2*medSpacing + (smallNum - 17)*smallSpacing, (largeNum-2)*largeSpacing)
    #Bottom right square
    smoothedMatrix = smoothRegion(smoothedMatrix, largeSpacing + 2*medSpacing + (smallNum - 17)*smallSpacing, (largeNum-2)*largeSpacing, largeSpacing + 2*medSpacing + (smallNum - 17)*smallSpacing, (largeNum-2)*largeSpacing)
    
    return smoothedMatrix
