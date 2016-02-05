# Andreas Vlachos, 2013:
# export PYTHONPATH="hvector/build/lib.linux-x86_64-2.7/:$PYTHONPATH"
from _mycollections import mydefaultdict
from mydouble import mydouble, counts
import cPickle as pickle
import gzip
from operator import itemgetter

import random
import math
import numpy

# cost-sensitive multiclass classification with AROW
# the instances consist of a dictionary of labels to costs and feature vectors (Huang-style)

class Instance(object):
    """
    An data instance to be used with AROW. Each instance is composed of a 
    feature vector (a dict or Huang-style sparse vector) and a dictionary
    of costs (where the labels should be encoded).
    """

    def __init__(self, featureVector, costs=None):
        self.featureVector = featureVector
        self.costs = costs
        if self.costs != None:
            self._normalize_costs()

    def _normalize_costs(self):
        """
        Normalize the costs by setting the lowest one to zero and the rest
        as increments over zero. 
        """
        min_cost = float("inf")
        self.maxCost = float("-inf")
        self.worstLabels = []
        self.correctLabels = []
        for label, cost in self.costs.items():
            if cost < min_cost:
                min_cost = cost
                self.correctLabels = [label]
            elif cost == min_cost:
                self.correctLabels.append(label)
            if cost > self.maxCost:
                self.maxCost = cost
                self.worstLabels = [label]
            elif cost == self.maxCost:
                self.worstLabels.append(label)
        if min_cost > 0:
            for label in self.costs:
                self.costs[label] -= min_cost
            self.maxCost -= min_cost

    def __str__(self):
        retString = ""
        labels = []
        for label,cost in self.costs.items():
            labels.append(label+":"+str(cost))
        retString += ",".join(labels)

        retString += "\t"
        
        features = []
        for feature in self.featureVector:
            features.append(feature + ":" + str(self.featureVector[feature]))
        
        retString += " ".join(features)

        return retString

    @staticmethod
    def removeHapaxLegomena(instances):
        print "Counting features"
        feature2counts = mydefaultdict(mydouble)
        for instance in instances:
            for element in instance.featureVector:
                feature2counts[element] += 1
        print len(feature2counts)

        print "Removing hapax legomena"
        newInstances = []
        for instance in instances:
            newFeatureVector = mydefaultdict(mydouble)
            for element in instance.featureVector:
                # if this feature was encountered more than once
                if feature2counts[element] > 1:
                    newFeatureVector[element] = instance.featureVector[element]
            newInstances.append(Instance(newFeatureVector, instance.costs))
        return newInstances


class Prediction:

    def __init__(self):
        self.label2score = {}
        self.score = float("-inf")
        self.label = None
        self.featureValueWeights = []
        self.label2prob = {}
        self.entropy = 0.0

class AROW():

    def __init__(self):
        self.probabilities = False
        self.currentWeightVectors = {}
        self.currentVarianceVectors = {}

    # This predicts always using the current weight vectors
    def predict(self, instance, verbose=False, probabilities=False):
        # always add the bias
        instance.featureVector["biasAutoAdded"] = 1.0

        prediction = Prediction()
        
        for label, weightVector in self.currentWeightVectors.items():
            score = instance.featureVector.dot(weightVector)
            prediction.label2score[label] = score
            if score > prediction.score:
                prediction.score = score
                prediction.label = label

        if verbose:
            for feature in instance.featureVector:
                # keep the feature weights for the predicted label
                prediction.featureValueWeights.append([feature, instance.featureVector[feature], self.currentWeightVectors[prediction.label][feature]])
            # order them from the most positive to the most negative
                prediction.featureValueWeights = sorted(prediction.featureValueWeights, key=itemgetter(2))
        if probabilities:
            # if we have probabilistic training
            if self.probabilities:
                probPredictions ={}
                for label in self.probWeightVectors[0].keys():
                    # smoothing the probabilities with add 0.01 of 1 out of the vectors
                    probPredictions[label] = 0.01/len(self.probWeightVectors)
                # for each of the weight vectors obtained get its prediction
                for probWeightVector in self.probWeightVectors:
                    maxScore = float("-inf")
                    maxLabel = None
                    for label, weightVector in probWeightVector.items():
                        score = instance.featureVector.dot(weightVector)
                        if score > maxScore:
                            maxScore = score
                            maxLabel = label
                    # so the winning label adds one vote
                    probPredictions[maxLabel] += 1

                # now let's normalize:
                for label, score in probPredictions.items():
                    prediction.label2prob[label] = float(score)/len(self.probWeightVectors)

                # Also compute the entropy:
                for prob in prediction.label2prob.values():
                    if prob > 0:
                        prediction.entropy -= prob * math.log(prob, 2)
                # normalize it:
                prediction.entropy /= math.log(len(prediction.label2prob),2)
                #print prediction.label2prob
                #print prediction.entropy
            else:
                print "Need to obtain weight samples for probability estimates first"

        return prediction

    # This is just used to optimize the params
    # if probabilities is True we return the ratio for the average entropies, otherwise the loss
    def batchPredict(self, instances, probabilities=False):
        totalCost = 0
        sumCorrectEntropies = 0
        sumIncorrectEntropies = 0
        sumLogProbCorrect = 0
        totalCorrects = 0
        totalIncorrects = 0
        sumEntropies = 0
        for instance in instances:
            prediction = self.predict(instance, False, probabilities)
            # This is without probabilities, with probabilities we want the average entropy*cost 
            if probabilities:
                if instance.costs[prediction.label] == 0:
                    sumLogProbCorrect -= math.log(prediction.label2prob[prediction.label],2)
                    totalCorrects += instance.maxCost
                    sumEntropies += instance.maxCost*prediction.entropy
                    sumCorrectEntropies += instance.maxCost*prediction.entropy
                else:
                    maxCorrectProb = 0.0
                    for correctLabel in instance.correctLabels:
                        if prediction.label2prob[correctLabel] > maxCorrectProb:
                            maxCorrectProb = prediction.label2prob[correctLabel]
                    #if maxCorrectProb > 0.0:
                    sumLogProbCorrect -= math.log(maxCorrectProb, 2)
                    #else:
                    #    sumLogProbCorrect = float("inf")
                    totalIncorrects += instance.maxCost
                    sumEntropies += instance.maxCost*(1-prediction.entropy)
                    sumIncorrectEntropies += instance.maxCost*prediction.entropy
                    
            else:
                # no probs, just keep track of the cost incurred
                if instance.costs[prediction.label] > 0:
                    totalCost += instance.costs[prediction.label]

        if probabilities:
            avgCorrectEntropy = sumCorrectEntropies/float(totalCorrects)
            print avgCorrectEntropy
            avgIncorrectEntropy = sumIncorrectEntropies/float(totalIncorrects)
            print avgIncorrectEntropy
            print sumLogProbCorrect
            return sumLogProbCorrect
        else:
            return totalCost

    # the parameter here is for AROW learning
    # adapt if True is AROW, if False it is passive aggressive-II with prediction-based updates 
    def train(self, instances, averaging=True, shuffling=True, rounds = 10, param = 1, adapt=True):
        # we first need to go through the dataset to find how many classes

        # Initialize the weight vectors in the beginning of training"
        # we have one variance and one weight vector per class
        self.currentWeightVectors = {} 
        if adapt:
            self.currentVarianceVectors = {}
        if averaging:
            averagedWeightVectors = {}
            updatesLeft = rounds*len(instances)
        for label in instances[0].costs:
            self.currentWeightVectors[label] = mydefaultdict(mydouble)
            # remember: this is sparse in the sense that everething that doesn't have a value is 1
            # everytime we to do something with it, remember to add 1
            if adapt:
                self.currentVarianceVectors[label] = {}
            # keep the averaged weight vector
            if averaging:
                averagedWeightVectors[label] = mydefaultdict(mydouble)

        # in each iteration        
        for r in range(rounds):
            # shuffle
            if shuffling:
                random.shuffle(instances)
            errorsInRound = 0
            costInRound = 0
            # for each instance
            for instance in instances:
                prediction = self.predict(instance)

                # so if the prediction was incorrect
                # we are no longer large margin, since we are using the loss from the cost-sensitive PA
                if instance.costs[prediction.label] > 0:
                    errorsInRound += 1
                    costInRound += instance.costs[prediction.label]

                    # first we need to get the score for the correct answer
                    # if the instance has more than one correct answer then pick the min
                    minCorrectLabelScore = float("inf")
                    minCorrectLabel = None
                    for label in instance.correctLabels:
                        score = instance.featureVector.dot(self.currentWeightVectors[label])
                        if score < minCorrectLabelScore:
                            minCorrectLabelScore = score
                            minCorrectLabel = label
                            
                    # the loss is the scaled margin loss also used by Mejer and Crammer 2010
                    loss = prediction.score - minCorrectLabelScore  + math.sqrt(instance.costs[prediction.label])
                        
                    if adapt:
                        # Calculate the confidence values
                        # first for the predicted label
                        zVectorPredicted = mydefaultdict(mydouble)
                        zVectorMinCorrect = mydefaultdict(mydouble)
                        for feature in instance.featureVector:
                            # the variance is either some value that is in the dict or just 1
                            if feature in self.currentVarianceVectors[prediction.label]:
                                zVectorPredicted[feature] = instance.featureVector[feature] * self.currentVarianceVectors[prediction.label][feature]
                            else:
                                zVectorPredicted[feature] = instance.featureVector[feature]
                            # then for the minCorrect:
                            if feature in self.currentVarianceVectors[minCorrectLabel]:
                                zVectorMinCorrect[feature] = instance.featureVector[feature] * self.currentVarianceVectors[minCorrectLabel][feature]
                            else:
                                zVectorMinCorrect[feature] = instance.featureVector[feature]
                    
                        confidence = zVectorPredicted.dot(instance.featureVector) + zVectorMinCorrect.dot(instance.featureVector)

                        beta = 1.0/(confidence + param)

                        alpha = loss * beta

                        # update the current weight vectors
                        self.currentWeightVectors[prediction.label].iaddc(zVectorPredicted, -alpha)
                        self.currentWeightVectors[minCorrectLabel].iaddc(zVectorMinCorrect, alpha)

                        if averaging:
                            averagedWeightVectors[prediction.label].iaddc(zVectorPredicted, -alpha * updatesLeft)
                            averagedWeightVectors[minCorrectLabel].iaddc(zVectorMinCorrect, alpha * updatesLeft)
                        
                    else:
                        # the squared norm is twice the square of the features since they are the same per class 
                        norm = 2*(instance.featureVector.dot(instance.featureVector))
                        factor = loss/(norm + float(1)/(2*param))
                        self.currentWeightVectors[prediction.label].iaddc(instance.featureVector, -factor)
                        self.currentWeightVectors[minCorrectLabel].iaddc(instance.featureVector, factor)

                        if averaging:
                            averagedWeightVectors[prediction.label].iaddc(instance.featureVector, -factor * updatesLeft)
                            averagedWeightVectors[minCorrectLabel].iaddc(instance.featureVector, factor * updatesLeft)
                        
                    
                    if adapt:
                        # update the diagonal covariance
                        for feature in instance.featureVector.iterkeys():
                            # for the predicted
			                if feature in self.currentVarianceVectors[prediction.label]:
			                    self.currentVarianceVectors[prediction.label][feature] -= beta * pow(zVectorPredicted[feature],2)
			                else:
			                    # Never updated this covariance before, add 1
			                    self.currentVarianceVectors[prediction.label][feature] = 1 - beta * pow(zVectorPredicted[feature],2)
                            # for the minCorrect
			                if feature in self.currentVarianceVectors[minCorrectLabel]:
			                    self.currentVarianceVectors[minCorrectLabel][feature] -= beta * pow(zVectorMinCorrect[feature],2)
			                else:
			                    # Never updated this covariance before, add 1
			                    self.currentVarianceVectors[minCorrectLabel][feature] = 1 - beta * pow(zVectorMinCorrect[feature],2)

                if averaging:
		            updatesLeft-=1
                
            print "Training error rate in round " + str(r) + " : " + str(float(errorsInRound)/len(instances))
	    
        if averaging:
            for label in self.currentWeightVectors:
                self.currentWeightVectors[label] = mydefaultdict(mydouble)
                self.currentWeightVectors[label].iaddc(averagedWeightVectors[label], 1.0/float(rounds*len(instances)))

        # Compute the final training error:
        finalTrainingErrors = 0
        finalTrainingCost = 0
        for instance in instances:
            prediction = self.predict(instance)
            if instance.costs[prediction.label] > 0:
                finalTrainingErrors +=1
                finalTrainingCost += instance.costs[prediction.label]

        finalTrainingErrorRate = float(finalTrainingErrors)/len(instances)
        print "Final training error rate=" + str(finalTrainingErrorRate)
        print "Final training cost=" + str(finalTrainingCost)

        return finalTrainingCost

    def probGeneration(self, scale=1.0, noWeightVectors=100):
        # initialize the weight vectors
        print "Generating samples for the weight vectors to obtain probability estimates"
        self.probWeightVectors = []
        for i in xrange(noWeightVectors):
            self.probWeightVectors.append({})
            for label in self.currentWeightVectors:
                self.probWeightVectors[i][label] = mydefaultdict(mydouble)

        for label in self.currentWeightVectors:
            # We are ignoring features that never got their weight set 
            for feature in self.currentWeightVectors[label]:
                # note that if the weight was updated, then the variance must have been updated too, i.e. we shouldn't have 0s
                weights = numpy.random.normal(self.currentWeightVectors[label][feature], scale * self.currentVarianceVectors[label][feature], noWeightVectors)
                # we got the samples, now let's put them in the right places
                for i,weight in enumerate(weights):
                    self.probWeightVectors[i][label][feature] = weight
                
        print "done"
        self.probabilities = True

    # train by optimizing the c parametr
    @staticmethod
    def trainOpt(instances, rounds = 10, paramValues=[0.01, 0.1, 1.0, 10, 100], heldout=0.2, adapt=True, optimizeProbs=False):
        print "Training with " + str(len(instances)) + " instances"

        # this value will be kept if nothing seems to work better
        bestParam = 1
        lowestCost = float("inf")
        bestClassifier = None
        trainingInstances = instances[:int(len(instances) * (1-heldout))]
        testingInstances = instances[int(len(instances) * (1-heldout)) + 1:]
        for param in paramValues:
            print "Training with param="+ str(param) + " on " + str(len(trainingInstances)) + " instances"
            # Keep the weight vectors produced in each round
            classifier = AROW()
            classifier.train(trainingInstances, True, True, rounds, param, adapt)
            print "testing on " + str(len(testingInstances)) + " instances"
            # Test on the dev for the weight vector produced in each round
            devCost = classifier.batchPredict(testingInstances)
            print "Dev cost:" + str(devCost) + " avg cost per instance " + str(devCost/float(len(testingInstances)))

            if devCost < lowestCost:
                bestParam = param
                lowestCost = devCost
                bestClassifier = classifier

        # OK, now we got the best C, so it's time to train the final model with it
        # Do the probs
        # So we need to pick a value between 
        if optimizeProbs:
            print "optimizing the scale parameter for probability estimation"
            bestScale = 1.0
            lowestEntropy = float("inf")
            steps = 20
            for i in xrange(steps):
                scale = 1.0 - float(i)/steps
                print "scale= " +  str(scale)
                bestClassifier.probGeneration(scale)
                entropy = bestClassifier.batchPredict(testingInstances, True)
                print "entropy sums: " + str(entropy)
                
                if entropy < lowestEntropy:
                    bestScale = scale
                    lowestEntropy = entropy
        
        
        # Now train the final model:
        print "Training with param="+ str(bestParam) + " on all the data"

        finalClassifier = AROW()
        finalClassifier.train(instances, True, True, rounds, bestParam, adapt)
        if optimizeProbs:
            print "Adding weight samples for probability estimates with scale " + str(bestScale)
            finalClassifier.probGeneration(bestScale)

        return finalClassifier
        
    # save function for the parameters:
    def save(self, filename):
        model_file = open(filename, "w")
        # prepare for pickling 
        pickleDict = {}
        for label in self.currentWeightVectors:
            pickleDict[label] = {}
            for feature in self.currentWeightVectors[label]:
                pickleDict[label][feature] = self.currentWeightVectors[label][feature]
        pickle.dump(pickleDict, model_file)
        model_file.close()
        # Check if there are samples for probability estimates to save
        if self.probabilities:
            pickleDictProbVectors = []
            for sample in self.probWeightVectors:
                label2vector = {}
                for label, vector in sample.items():
                    label2vector[label] = {}
                    for feature in vector:
                        label2vector[label][feature] = vector[feature]
                pickleDictProbVectors.append(label2vector)
            probVectorFile = gzip.open(filename + "_probVectors.gz", "wb")
            pickle.dump(pickleDictProbVectors, probVectorFile, -1)
            probVectorFile.close()
        # this is just for debugging, doesn't need to be loaded as it is not used for prediction
        # Only the non-one variances are added
        pickleDictVar = {}
        covariance_file = open(filename + "_variances", "w")
        for label in self.currentVarianceVectors:
            pickleDictVar[label] = {}
            for feature in self.currentVarianceVectors[label]:
                pickleDictVar[label][feature] = self.currentVarianceVectors[label][feature]
        pickle.dump(pickleDictVar, covariance_file)
        covariance_file.close()


    # load a model from a file:
    def load(self, filename):
        model_weights = open(filename, 'r')
        weightVectors = pickle.load(model_weights)
        model_weights.close()
        for label, weightVector in weightVectors.items():
            self.currentWeightVectors[label] = mydefaultdict(mydouble, weightVector)

        try:
            with gzip.open(filename + "_probVectors.gz", "rb") as probFile:
                print "loading probabilities"
                pickleDictProbVectors = pickle.load(probFile)
                self.probWeightVectors = []
                for sample in pickleDictProbVectors:
                    label2Vectors = {}
                    for label,vector in sample.items():
                        label2Vectors[label] = mydefaultdict(mydouble, vector)
                    self.probWeightVectors.append(label2Vectors)

                probFile.close()
                self.probabilities = True
        except IOError:
            print 'No weight vectors for probability estimates'
            self.probabilities = False
        
    

if __name__ == "__main__":

    import sys
    import random
    random.seed(13)           
    numpy.random.seed(13)
    dataLines = open(sys.argv[1]).readlines()

    instances = []
    classifier_p = AROW()
    print "Reading the data"
    for line in dataLines:
        details = line.split()
        costs = {}
        featureVector = mydefaultdict(mydouble)
        
        if details[0] == "-1":
            costs["neg"] = 0
            costs["pos"] = 1
        elif details[0] == "+1":
            costs["neg"] = 1
            costs["pos"] = 0

        for feature in details[1:]:
            featureID, featureVal = feature.split(":")
            featureVector[featureID] = float(featureVal)
            #featureVector["dummy"+str(len(instances))] = 1.0
            #featureVector["dummy2"+str(len(instances))] = 1.0
            #featureVector["dummy3"+str(len(instances))] = 1.0
        instances.append(Instance(featureVector, costs))
        #print instances[-1].costs

    random.shuffle(instances)
    #instances = instances[:100]
    # Keep some instances to check the performance
    testingInstances = instances[int(len(instances) * 0.75) + 1:]
    trainingInstances = instances[:int(len(instances) * 0.75)]

    print "training data: " + str(len(trainingInstances)) + " instances"
    #trainingInstances = Instance.removeHapaxLegomena(trainingInstances)
    #classifier_p.train(trainingInstances, True, True, 10, 0.1, False)
    
    # the penultimate parameter is True for AROW, false for PA
    # the last parameter can be set to True if probabilities are needed.
    classifier_p = AROW.trainOpt(trainingInstances, 10, [0.01, 0.1, 1.0, 10, 100], 0.1, True, False)

    cost = classifier_p.batchPredict(testingInstances)
    avgCost = float(cost)/len(testingInstances)
    print "Avg Cost per instance " + str(avgCost) + " on " + str(len(testingInstances)) + " testing instances"

    #avgRatio = classifier_p.batchPredict(testingInstances, True)
    #print "entropy sums: " + str(avgRatio)

    # Save the parameters:
    #print "saving"
    #classifier_p.save(sys.argv[1] + ".arow")    
    #print "done"
    # load again:
    #classifier_new = AROW()
    #print "loading model"
    #classifier_new.load(sys.argv[1] + ".arow")
    #print "done"

    #avgRatio = classifier_new.batchPredict(testingInstances, True)
    #print "entropy sums: " + str(avgRatio)
