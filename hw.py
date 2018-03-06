
import numpy as np
from numpy.linalg import norm
import math
import random


def calUserSimilarity(inputTrain, inputTest):
	maxlen = len(inputTrain)
	arr = [0] * maxlen
	for i in range(0, maxlen):
		distance1 = 0
		distance2 = 0
		for j in range(0, len(inputTrain[0])):
			if inputTest[j] != 0 and inputTrain[i][j] != 0:
				distance1 = distance1 + inputTrain[i][j] * inputTrain[i][j]
				distance2 = distance2 + inputTest[j] * inputTest[j]
		if(distance1 == 0 or distance2 == 0):
			arr[i] = 0
		else:
			arr[i] = np.dot(inputTrain[i], inputTest) / (math.sqrt(distance1) * math.sqrt(distance2))
	return arr


#do not ignore zeros
def calNormalCosineSimilarity(inputTrain, inputTest):
	maxlen = len(inputTrain)
	arr = np.zeros(maxlen)
	inputTestDist = np.linalg.norm(inputTest)
	for i in range(0, maxlen):
		denominator = np.linalg.norm(inputTrain[i]) * inputTestDist
		if denominator != 0:
			arr[i] = np.dot(inputTrain[i], inputTest) / denominator
	return arr


def computeCosineRating(inputTrain, k, testRating, testUnknown):
	ratingReturn = []
	for i in range(0, len(testRating)):
		testUid = testRating[i]
		weightArray = calUserSimilarity(inputTrain, testUid)	
		weightIndexArray = np.argsort(weightArray, axis = 0 )[::-1]
		ratingarr = []
		for j in range(0, len(testUnknown[i])):
			mid = testUnknown[i][j]
			count = 0
			numerator = 0
			denominator = 0
			for a in range(0, len(weightIndexArray)):
				uid = weightIndexArray[a]				
				if inputTrain[uid][mid - 1] != 0:
					numerator += weightArray[uid] * inputTrain[uid][mid - 1]
					denominator += weightArray[uid]
					count += 1
					if count == k:
						break
			if denominator == 0:
				ratingarr.append(0)
			else:
				ratingarr.append(numerator / denominator)
		ratingReturn.append(ratingarr)

	return ratingReturn



def transformToAvgMatrix(inputTrain):
	newTrain = np.zeros((len(inputTrain), len(inputTrain[0])))
	avgArr = np.zeros(len(inputTrain))
	for i in range(0, len(inputTrain)):
		sum = 0
		count = 0
		for j in range(0, len(inputTrain[0])):
			if inputTrain[i][j] != 0:
				sum = sum + inputTrain[i][j]
				count = count + 1
		if count != 0:
			avgArr[i] = sum / count

		for k in range(0, len(inputTrain[0])):
			if inputTrain[i][k] != 0:
				newTrain[i][k] = inputTrain[i][k] - avgArr[i]
	return newTrain, avgArr


def computePearsonCorrRating(inputTrain, k, testKnown, testUnknown):
	avgTrain, avgArrTrain = transformToAvgMatrix(inputTrain)
	avgTestKnown, avgArrTest = transformToAvgMatrix(testKnown)
	ratingReturn = []
		
	for i in range(0, len(testKnown)):

		weightArray = calUserSimilarity(avgTrain, avgTestKnown[i])
		absArr = np.zeros(len(weightArray))
		for b in range(0, len(weightArray)):
			absArr[b] = abs(weightArray[b])
		weightIndexArray = np.argsort(absArr, axis = 0 )[::-1]
		ratingarr = []
		
		for j in range(0, len(testUnknown[i])):
			colid = testUnknown[i][j]
			count = 0
			numerator = 0
			denominator = 0
			for a in range(0, len(weightIndexArray)):
				rowid = weightIndexArray[a]				
				if inputTrain[rowid][colid - 1] != 0:
					numerator += weightArray[rowid] * avgTrain[rowid][colid - 1]
					denominator += absArr[rowid]
					count += 1
					if count == k:
						break
			if denominator == 0:
				ratingarr.append(0)
			else:
				ratingarr.append(avgArrTest[i] + numerator / denominator)
		ratingReturn.append(ratingarr)

	return ratingReturn


def computeIuf(inputTrain):
	iufArr = np.zeros(len(inputTrain[0]))	
	for i in range(0, len(inputTrain[0])):
		count = 0
		for j in range(0, len(inputTrain)):
			if inputTrain[j][i] != 0:
				count += 1
		if count != 0:
			iufArr[i] = math.log(len(inputTrain) / count)
	return iufArr


def transformToIufMatrix(inputTrain, iufArr):
	newTrain = np.zeros((len(inputTrain), len(inputTrain[0])))
	for i in range(0, len(inputTrain[0])):
		for j in range(0, len(inputTrain)):
			newTrain[j][i] = inputTrain[j][i] * iufArr[i]
	return newTrain


def computeIufPearsonCorrRating(inputTrain, k, testKnown, testUnknown):
	avgTrain, avgArrTrain = transformToAvgMatrix(inputTrain)
	avgTestKnown, avgArrTest = transformToAvgMatrix(testKnown)
	iufArr = computeIuf(inputTrain)
	iufTrain = transformToIufMatrix(avgTrain, iufArr)
	iufTestKnown = transformToIufMatrix(avgTestKnown, iufArr)

	
	ratingReturn = []	
	for i in range(0, len(testKnown)):

		weightArray = calUserSimilarity(iufTrain, iufTestKnown[i])
		absArr = np.zeros(len(weightArray))
		for b in range(0, len(weightArray)):
			absArr[b] = abs(weightArray[b])
		weightIndexArray = np.argsort(absArr, axis = 0 )[::-1]

		ratingarr = []
		
		for j in range(0, len(testUnknown[i])):
			colid = testUnknown[i][j]
			count = 0
			numerator = 0
			denominator = 0
			for a in range(0, len(weightIndexArray)):
				rowid = weightIndexArray[a]				
				if inputTrain[rowid][colid - 1] != 0:
					numerator += weightArray[rowid] * avgTrain[rowid][colid - 1]
					denominator += absArr[rowid]
					count += 1
					if count == k:
						break
			if denominator == 0:
				ratingarr.append(0)
			else:
				ratingarr.append(avgArrTest[i] + numerator / denominator)

		ratingReturn.append(ratingarr)

	return ratingReturn


def computeCasePearsonCorrRating(inputTrain, k, testKnown, testUnknown):
	avgTrain, avgArrTrain = transformToAvgMatrix(inputTrain)
	avgTestKnown, avgArrTest = transformToAvgMatrix(testKnown)
	ratingReturn = []	
	for i in range(0, len(testKnown)):

		weightArray = calUserSimilarity(avgTrain, avgTestKnown[i])
		absArr = np.zeros(len(weightArray))
		for b in range(0, len(weightArray)):
			weightArray[b] = math.pow(abs(weightArray[b]), 1.5) * weightArray[b]
			absArr[b] = weightArray[b]
		weightIndexArray = np.argsort(absArr, axis = 0 )[::-1]

		ratingarr = []
		
		for j in range(0, len(testUnknown[i])):
			colid = testUnknown[i][j]
			count = 0
			numerator = 0
			denominator = 0
			for a in range(0, len(weightIndexArray)):
				rowid = weightIndexArray[a]				
				if inputTrain[rowid][colid - 1] != 0:
					numerator += weightArray[rowid] * avgTrain[rowid][colid - 1]
					denominator += absArr[rowid]
					count += 1
					if count == k:
						break
			if denominator == 0:
				ratingarr.append(0)
			else:
				ratingarr.append(avgArrTest[i] + numerator / denominator)

		ratingReturn.append(ratingarr)

	return ratingReturn



def helperItemBase(inputTrain):
	avgTrain, avgArrTrain = transformToAvgMatrix(inputTrain)
	transTrain = np.transpose(avgTrain)
	movieMatrix = []
	weightMatrix = []
	
	for i in range(len(transTrain)):
		weightArray = calNormalCosineSimilarity(transTrain, transTrain[i])
		absArr = np.zeros(len(weightArray))
		for b in range(0, len(weightArray)):
			absArr[b] = abs(weightArray[b])
		weightIndexArray = np.argsort(absArr, axis = 0 )[::-1]
		movieMatrix.append(weightIndexArray)
		weightMatrix.append(weightArray)
	return movieMatrix, weightMatrix




'''
def computeItemBaseRating(movieMatrix, weightMatrix, k, testKnown, testUnknown):
	ratingReturn = []
	for i in range(0, len(testUnknown)):
		ratingarr = []
		for j in range(0, len(testUnknown[i])):
			count = 0
			numerator = 0
			denominator = 0
			mid = testUnknown[i][j]	- 1
			for c in range(0, len(movieMatrix[mid])):
				checkmid = movieMatrix[mid][c]
				if testKnown[i][checkmid] != 0:
					numerator += weightMatrix[mid][checkmid] * testKnown[i][checkmid]
					denominator += weightMatrix[mid][checkmid]
					count += 1
					if count == k:
						break
			if denominator == 0:
				ratingarr.append(0)
			else:
				ratingarr.append(numerator / denominator)
		ratingReturn.append(ratingarr)	

	return ratingReturn	

'''

def computeNokItemBaseRating(movieMatrix, weightMatrix, testKnown, testUnknown):
	ratingReturn = []
	for i in range(0, len(testUnknown)):
		ratingarr = []
		for j in range(0, len(testUnknown[i])):
			numerator = 0
			denominator = 0
			mid = testUnknown[i][j]	- 1
			for c in range(0, len(movieMatrix[mid])):
				checkmid = movieMatrix[mid][c]
				if testKnown[i][checkmid] != 0 and abs(weightMatrix[mid][checkmid]) >= 0.6:
					numerator += weightMatrix[mid][checkmid] * testKnown[i][checkmid]
					denominator += weightMatrix[mid][checkmid]
			if denominator == 0:
				ratingarr.append(0)
			else:
				ratingarr.append(numerator / denominator)
		ratingReturn.append(ratingarr)	
	return ratingReturn	

'''
def helperOwnItemBase(inputTrain):
	avgTrain, avgArrTrain = transformToAvgMatrix(inputTrain)
	transTrain = np.transpose(avgTrain)
	arr = computeIuf(transTrain)
	newTrain = transformToIufMatrix(transTrain, arr)
	movieMatrix = []
	weightMatrix = []
	
	for i in range(len(newTrain)):
		weightArray = calNormalCosineSimilarity(newTrain, newTrain[i])
		#weightArray = calUserSimilarity(transTrain, transTrain[i])
		absArr = np.zeros(len(weightArray))
		for b in range(0, len(weightArray)):
			absArr[b] = abs(weightArray[b])
		weightIndexArray = np.argsort(absArr, axis = 0 )[::-1]
		movieMatrix.append(weightIndexArray)
		weightMatrix.append(weightArray)
	return movieMatrix, weightMatrix


def computeOwn(movieMatrix, weightMatrix, testKnown, testUnknown):
	ratingReturn = []
	for i in range(0, len(testUnknown)):
		ratingarr = []
		for j in range(0, len(testUnknown[i])):
			numerator = 0
			denominator = 0
			mid = testUnknown[i][j]	- 1
			for c in range(0, len(movieMatrix[mid])):
				checkmid = movieMatrix[mid][c]
				if testKnown[i][checkmid] != 0 and abs(weightMatrix[mid][checkmid]) >= 0.6:
					numerator += weightMatrix[mid][checkmid] * testKnown[i][checkmid]
					denominator += weightMatrix[mid][checkmid]
			if denominator == 0:
				ratingarr.append(0)
			else:
				ratingarr.append(numerator / denominator)
		ratingReturn.append(ratingarr)	
	return ratingReturn	

'''

def calOwnSimi(inputTrain, inputTest):
	maxlen = len(inputTrain)
	arr = [0] * maxlen
	for i in range(0, maxlen):	
		commonCount = 0
		count1 = 0
		count2 = 0
		for k in range(0, len(inputTrain[0])):
			if inputTrain[i][k] != 0:
				count1 += 1
			if inputTest[k] != 0:
				count2 += 1
			if inputTest[k] != 0 and inputTrain[i][k] != 0:
				commonCount += 1

		dis = 0	
		numerator = 0
		for j in range(0, len(inputTrain[0])):
			if inputTest[j] != 0 and inputTrain[i][j] != 0:
				dis = 2 - abs(inputTest[j] - inputTrain[i][j])
				numerator += dis / math.log(1 + commonCount)	
		arr[i] = numerator / (math.sqrt(count1) * math.sqrt(count2))
	return arr

def computeOwnRating(inputTrain, k, testRating, testUnknown):
	ratingReturn = []
	for i in range(0, len(testRating)):
		testUid = testRating[i]
		weightArray = calOwnSimi(inputTrain, testUid)	
		weightIndexArray = np.argsort(weightArray, axis = 0 )[::-1]
		ratingarr = []
		for j in range(0, len(testUnknown[i])):
			mid = testUnknown[i][j]
			count = 0
			numerator = 0
			denominator = 0
			for a in range(0, len(weightIndexArray)):
				uid = weightIndexArray[a]				
				if inputTrain[uid][mid - 1] != 0:
					numerator += weightArray[uid] * inputTrain[uid][mid - 1]
					denominator += weightArray[uid]
					count += 1
					if count == k:
						break
			if denominator == 0:
				ratingarr.append(0)
			else:
				ratingarr.append(numerator / denominator)
		ratingReturn.append(ratingarr)

	return ratingReturn



def transformTestData(testRaw, movieLen):
	arr = []
	prevNum = -1
	matrix1 = []
	matrix2 = []

	for i in range(0, len(testRaw)):
		if testRaw[i][0] != prevNum:
			arr.append(testRaw[i][0])
			prevNum = testRaw[i][0]
			matrix1.append(np.zeros(movieLen))
			matrix2.append([])
		
		dynamicArrayLen = len(arr)
		mid = testRaw[i][1]
		movieRate = testRaw[i][2]
		if movieRate != 0:
			matrix1[dynamicArrayLen - 1][mid - 1] = movieRate

		if movieRate == 0:
			matrix2[dynamicArrayLen - 1].append(mid)

	return arr, matrix1, matrix2



def writeToFinalFile(uidArr, matrixKnown, matrixUnknown, resultRatingMatrix, fileName):
	outFile = open(fileName, 'w')
	for i in range(0, len(uidArr)):
		sum = 0
		count = 0
		avg = 3
		for j in range(len(matrixKnown[0])):
			if matrixKnown[i][j] != 0:
				sum = sum + matrixKnown[i][j]
				count = count + 1
				#outFile.write('{} {} {}\n'.format(uidArr[i], j + 1, int(matrixKnown[i][j])))
		if count != 0:
			avg = sum / count
		for j in range(len(matrixUnknown[i])):
			normalizeResult = int(round(avg))
			if resultRatingMatrix[i][j] != 0:
				if resultRatingMatrix[i][j] <= 1:
					normalizeResult = 1
				elif resultRatingMatrix[i][j] >= 5:
					normalizeResult = 5
				else:
					normalizeResult = int(round(resultRatingMatrix[i][j]))

			outFile.write('{} {} {}\n'.format(uidArr[i], matrixUnknown[i][j], normalizeResult))

	outFile.close()


def helperCosineSimi(trainFile, testFile, outputFile):
	trainData = np.loadtxt(trainFile, dtype='i', delimiter='\t')
	testData =  np.loadtxt(testFile, dtype='i', delimiter=' ')
	movieLen = len(trainData[0])
	transformTestDataArr, transformTestDataMatrix1, transformTestDataMatrix2 = transformTestData(testData, movieLen)

	result = computeCosineRating(trainData, 40, transformTestDataMatrix1, transformTestDataMatrix2)

	writeToFinalFile(transformTestDataArr, transformTestDataMatrix1, transformTestDataMatrix2, result, outputFile)


def helperPearsonCorr(trainFile, testFile, outputFile):
	trainData = np.loadtxt(trainFile, dtype='i', delimiter='\t')
	testData =  np.loadtxt(testFile, dtype='i', delimiter=' ')
	movieLen = len(trainData[0])
	transformTestDataArr, transformTestDataMatrix1, transformTestDataMatrix2 = transformTestData(testData, movieLen)

	result = computePearsonCorrRating(trainData, 50, transformTestDataMatrix1, transformTestDataMatrix2)

	writeToFinalFile(transformTestDataArr, transformTestDataMatrix1, transformTestDataMatrix2, result, outputFile)


def helperIufPearsonCorr(trainFile, testFile, outputFile):
	trainData = np.loadtxt(trainFile, dtype='i', delimiter='\t')
	testData =  np.loadtxt(testFile, dtype='i', delimiter=' ')
	movieLen = len(trainData[0])
	transformTestDataArr, transformTestDataMatrix1, transformTestDataMatrix2 = transformTestData(testData, movieLen)

	result = computeIufPearsonCorrRating(trainData, 40, transformTestDataMatrix1, transformTestDataMatrix2)

	writeToFinalFile(transformTestDataArr, transformTestDataMatrix1, transformTestDataMatrix2, result, outputFile)



def helperCasePearsonCorr(trainFile, testFile, outputFile):
	trainData = np.loadtxt(trainFile, dtype='i', delimiter='\t')
	testData =  np.loadtxt(testFile, dtype='i', delimiter=' ')
	movieLen = len(trainData[0])
	transformTestDataArr, transformTestDataMatrix1, transformTestDataMatrix2 = transformTestData(testData, movieLen)

	result = computeCasePearsonCorrRating(trainData, 10, transformTestDataMatrix1, transformTestDataMatrix2)

	writeToFinalFile(transformTestDataArr, transformTestDataMatrix1, transformTestDataMatrix2, result, outputFile)


def helperItemBaseRating(trainFile):
	trainData = np.loadtxt(trainFile, dtype='i', delimiter='\t')
	test5Data =  np.loadtxt("test5.txt", dtype='i', delimiter=' ')
	test10Data =  np.loadtxt("test10.txt", dtype='i', delimiter=' ')
	test20Data =  np.loadtxt("test20.txt", dtype='i', delimiter=' ')
	movieLen = len(trainData[0])
	transformTest5DataArr, transformTest5DataMatrix1, transformTest5DataMatrix2 = transformTestData(test5Data, movieLen)
	transformTest10DataArr, transformTest10DataMatrix1, transformTest10DataMatrix2 = transformTestData(test10Data, movieLen)
	transformTest20DataArr, transformTest20DataMatrix1, transformTest20DataMatrix2 = transformTestData(test20Data, movieLen)

	movieMatrix, weightMatrix = helperItemBase(trainData)
	result5 = computeNokItemBaseRating(movieMatrix, weightMatrix, transformTest5DataMatrix1, transformTest5DataMatrix2)
	result10 = computeNokItemBaseRating(movieMatrix, weightMatrix, transformTest10DataMatrix1, transformTest10DataMatrix2)
	result20 = computeNokItemBaseRating(movieMatrix, weightMatrix, transformTest20DataMatrix1, transformTest20DataMatrix2)

	writeToFinalFile(transformTest5DataArr, transformTest5DataMatrix1, transformTest5DataMatrix2, result5,'reportFile5_itembase.txt')
	writeToFinalFile(transformTest10DataArr, transformTest10DataMatrix1, transformTest10DataMatrix2, result10, 'reportFile10_itembase.txt')
	writeToFinalFile(transformTest20DataArr, transformTest20DataMatrix1, transformTest20DataMatrix2, result20, 'reportFile20_itembase.txt')
	'''
	ownTest(np.loadtxt("train.txt", dtype='i', delimiter='\t'), 20)
	test20Data =  np.loadtxt("yytest20.txt", dtype='i', delimiter=' ')
	movieLen = len(trainData[0])
	transformTest20DataArr, transformTest20DataMatrix1, transformTest20DataMatrix2 = transformTestData(test20Data, movieLen)
	movieMatrix, weightMatrix = helperItemBase(trainData)
	#result20 = computeItemBaseRating(movieMatrix, weightMatrix, 10, transformTest20DataMatrix1, transformTest20DataMatrix2)
	result20 = computeNokItemBaseRating(movieMatrix, weightMatrix, transformTest20DataMatrix1, transformTest20DataMatrix2)
	writeToFinalFile(transformTest20DataArr, transformTest20DataMatrix1, transformTest20DataMatrix2, result20, 'yytest20_itembase.txt')
	validate('yytest20_itembase.txt', 'yytest20_res.txt')
	'''

	'''
def helperOwnAlgoRating(trainFile):
	trainData = np.loadtxt(trainFile, dtype='i', delimiter='\t')

	test5Data =  np.loadtxt("test5.txt", dtype='i', delimiter=' ')
	test10Data =  np.loadtxt("test10.txt", dtype='i', delimiter=' ')
	test20Data =  np.loadtxt("test20.txt", dtype='i', delimiter=' ')
	movieLen = len(trainData[0])
	transformTest5DataArr, transformTest5DataMatrix1, transformTest5DataMatrix2 = transformTestData(test5Data, movieLen)
	transformTest10DataArr, transformTest10DataMatrix1, transformTest10DataMatrix2 = transformTestData(test10Data, movieLen)
	transformTest20DataArr, transformTest20DataMatrix1, transformTest20DataMatrix2 = transformTestData(test20Data, movieLen)

	movieMatrix, weightMatrix = helperOwnItemBase(trainData)
	result5 = computeOwn(movieMatrix, weightMatrix, transformTest5DataMatrix1, transformTest5DataMatrix2)
	result10 = computeOwn(movieMatrix, weightMatrix, transformTest10DataMatrix1, transformTest10DataMatrix2)
	result20 = computeOwn(movieMatrix, weightMatrix, transformTest20DataMatrix1, transformTest20DataMatrix2)

	writeToFinalFile(transformTest5DataArr, transformTest5DataMatrix1, transformTest5DataMatrix2, result5,'reportFile5_ownAlgo.txt')
	writeToFinalFile(transformTest10DataArr, transformTest10DataMatrix1, transformTest10DataMatrix2, result10, 'reportFile10_ownAlgo.txt')
	writeToFinalFile(transformTest20DataArr, transformTest20DataMatrix1, transformTest20DataMatrix2, result20, 'reportFile20_ownAlgo.txt')
	

	ownTest(np.loadtxt("train.txt", dtype='i', delimiter='\t'), 20)
	test20Data =  np.loadtxt("yytest20.txt", dtype='i', delimiter=' ')
	movieLen = len(trainData[0])
	transformTest20DataArr, transformTest20DataMatrix1, transformTest20DataMatrix2 = transformTestData(test20Data, movieLen)
	movieMatrix, weightMatrix = helperOwnItemBase(trainData)
	result20 = computeOwn(movieMatrix, weightMatrix, transformTest20DataMatrix1, transformTest20DataMatrix2)
	writeToFinalFile(transformTest20DataArr, transformTest20DataMatrix1, transformTest20DataMatrix2, result20, 'yytest20_own.txt')
	validate('yytest20_own.txt', 'yytest20_res.txt')
	'''

def helperOwnAlgoRating(trainFile, testFile, outputFile):
	trainData = np.loadtxt(trainFile, dtype='i', delimiter='\t')
	testData =  np.loadtxt(testFile, dtype='i', delimiter=' ')
	movieLen = len(trainData[0])
	transformTestDataArr, transformTestDataMatrix1, transformTestDataMatrix2 = transformTestData(testData, movieLen)

	result = computeOwnRating(trainData, 40, transformTestDataMatrix1, transformTestDataMatrix2)

	writeToFinalFile(transformTestDataArr, transformTestDataMatrix1, transformTestDataMatrix2, result, outputFile)



def ownTrain(inputTrain):
	newTrain = []
	for i in range(0, 150):
		arr = []
		for j in range(0, len(inputTrain[0])):
			arr.append(inputTrain[i][j])
		newTrain.append(arr)

	np.savetxt("yytrain.txt", newTrain, delimiter="\t", fmt='%i')



def ownTest(inputTrain, knownSize):
	newTest = []
	newTestResult = []
	for i in range(150, len(inputTrain)):
		count = np.count_nonzero(inputTrain[i])
		if count > knownSize:
			knownArr = []
			nzCount = 0
			for k in range(0, len(inputTrain[0])):
				if inputTrain[i][k] != 0:
					if nzCount < knownSize:
						knownArr.append(k)
					else:
						pick = random.randint(0, nzCount)
						if pick < knownSize:
							knownArr[pick] = k
					nzCount += 1
			for k in knownArr:
				newTest.append([i + 1, k + 1, inputTrain[i][k]])
			knownSet = set(knownArr)
			for k in range(0, len(inputTrain[0])):
				if inputTrain[i][k] != 0 and k not in knownSet:
					newTest.append([i + 1, k + 1, 0])
					newTestResult.append([i + 1, k + 1, inputTrain[i][k]])
	np.savetxt("yytest" + str(knownSize) + ".txt", newTest, delimiter=" ", fmt='%i')
	np.savetxt("yytest" + str(knownSize) + "_res.txt", newTestResult, delimiter=" ", fmt='%i')

def validate(predictFile, resultFile): 
	predictData = np.loadtxt(predictFile, dtype='i', delimiter=' ')
	resultData = np.loadtxt(resultFile, dtype='i', delimiter=' ')
	sumSq = 0
	sumAbs = 0
	for i in range(len(predictData)):
		sumSq += (predictData[i][2] - resultData[i][2]) * (predictData[i][2] - resultData[i][2])
		sumAbs += abs(predictData[i][2] - resultData[i][2])
	print("rmse: " + str(math.sqrt(sumSq / len(predictData))))
	print("mae: " + str(sumAbs / len(predictData)))

def stats(inputTrain):
	for i in range(0, len(inputTrain)):
		print(np.count_nonzero(inputTrain[i]))

def main():
	'''
	trainData = np.loadtxt("train.txt", dtype='i', delimiter='\t')
	test5Data =  np.loadtxt("test5.txt", dtype='i', delimiter=' ')
	

	movieLen = len(trainData[0])
	transformTest5DataArr, transformTest5DataMatrix1, transformTest5DataMatrix2= transformTestData(test5Data, movieLen)

	result = computePearsonCorrRating(trainData, 10, transformTest5DataMatrix1, transformTest5DataMatrix2)
	helperCosineSimi('train.txt', 'test5.txt', 'reportFile5_cosineSimi.txt')
	helperCosineSimi('train.txt', 'test10.txt', 'reportFile10_cosineSimi.txt')
	helperCosineSimi('train.txt', 'test20.txt', 'reportFile20_cosineSimi.txt')



	helperPearsonCorr('train.txt', 'test5.txt', 'reportFile5_pearsonCorr.txt')
	helperPearsonCorr('train.txt', 'test10.txt', 'reportFile10_pearsonCorr.txt')
	helperPearsonCorr('train.txt', 'test20.txt', 'reportFile20_pearsonCorr.txt')
	

	trainData = np.loadtxt("train.txt", dtype='i', delimiter='\t')
	test5Data =  np.loadtxt("test5.txt", dtype='i', delimiter=' ')
	movieLen = len(trainData[0])
	transformTest5DataArr, transformTest5DataMatrix1, transformTest5DataMatrix2 = transformTestData(test5Data, movieLen)

	result = computeCasePearsonCorrRating(trainData, 10, transformTest5DataMatrix1, transformTest5DataMatrix2)
	print(result[0])
	
	
	helperIufPearsonCorr('train.txt', 'test5.txt', 'reportFile5_iuf.txt')
	helperIufPearsonCorr('train.txt', 'test10.txt', 'reportFile10_iuf.txt')
	helperIufPearsonCorr('train.txt', 'test20.txt', 'reportFile20_iuf.txt')
	

	helperCasePearsonCorr('train.txt', 'test5.txt', 'reportFile5_itembase.txt')
	helperCasePearsonCorr('train.txt', 'test10.txt', 'reportFile10_itembase.txt')
	helperCasePearsonCorr('train.txt', 'test20.txt', 'reportFile20_itembase.txt')

	helperCasePearsonCorr('train.txt', 'test5.txt', 'reportFile5_ownAlgo.txt')
	helperCasePearsonCorr('train.txt', 'test10.txt', 'reportFile10_ownAlgo.txt')
	helperCasePearsonCorr('train.txt', 'test20.txt', 'reportFile20_ownAlgo.txt')

	helperIufPearsonCorr('train.txt', 'test5.txt', 'reportFile5_iuf.txt')
	helperIufPearsonCorr('train.txt', 'test10.txt', 'reportFile10_iuf.txt')
	helperIufPearsonCorr('train.txt', 'test20.txt', 'reportFile20_iuf.txt')

	helperCasePearsonCorr('train.txt', 'test5.txt', 'reportFile5_case.txt')
	helperCasePearsonCorr('train.txt', 'test10.txt', 'reportFile10_case.txt')
	helperCasePearsonCorr('train.txt', 'test20.txt', 'reportFile20_case.txt')
	

	trainData = np.loadtxt("train.txt", dtype='i', delimiter='\t')
	test5Data =  np.loadtxt("test5.txt", dtype='i', delimiter=' ')
	movieLen = len(trainData[0])
	transformTest5DataArr, transformTest5DataMatrix1, transformTest5DataMatrix2 = transformTestData(test5Data, movieLen)

	result = computeItemBaseRating(trainData, 10, transformTest5DataMatrix1, transformTest5DataMatrix2)
	
	
	helperItemBaseRating('train.txt')
	'''
	
	#helperCosineSimi('train.txt', 'test5.txt', 'reportFile5_pearsonCorr.txt')
	'''
	helperIufPearsonCorr('train.txt', 'test5.txt', 'reportFile5_iuf.txt')
	helperIufPearsonCorr('train.txt', 'test10.txt', 'reportFile10_iuf.txt')
	helperIufPearsonCorr('train.txt', 'test20.txt', 'reportFile20_iuf.txt')
	'''
	#trainData = np.loadtxt("train.txt", dtype='i', delimiter='\t')
	#ownTrain(trainData)
	
	#stats(trainData

	'''
	ownTest(trainData, 5)
	helperPearsonCorr('yytrain.txt', 'yytest5.txt', 'yytest5_pearsonCorr.txt')
	validate('yytest5_pearsonCorr.txt', 'yytest5_res.txt')
	
	
	ownTest(trainData, 10)
	helperPearsonCorr('yytrain.txt', 'yytest10.txt', 'yytest10_pearsonCorr.txt')
	validate('yytest10_pearsonCorr.txt', 'yytest10_res.txt')
	
	
	ownTest(trainData, 20)
	helperCasePearsonCorr('yytrain.txt', 'yytest20.txt', 'yytest20_case.txt')
	validate('yytest20_case.txt', 'yytest20_res.txt')
	
	
	helperItemBaseRating('yytrain.txt')
	
	
	trainData = np.loadtxt("train.txt", dtype='i', delimiter='\t')
	helperItemBaseRating('train.txt')
	'''
	'''
	helperOwnAlgoRating('train.txt', 'test5.txt', 'reportFile5_iuf.txt')
	helperOwnAlgoRating('train.txt', 'test10.txt', 'reportFile10_iuf.txt')
	helperOwnAlgoRating('train.txt', 'test20.txt', 'reportFile20_iuf.txt')
	
	helperOwnAlgoRating('yytrain.txt')
	'''

	trainData = np.loadtxt("train.txt", dtype='i', delimiter='\t')
	ownTest(trainData, 20)
	helperOwnAlgoRating('yytrain.txt', 'yytest20.txt', 'yytest20_own.txt')
	validate('yytest20_own.txt', 'yytest20_res.txt')
	
if __name__== "__main__":
  	main()
