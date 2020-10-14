# Copyright 2015-2016 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np

from joblib import dump, load
from sklearn.model_selection import train_test_split

from .data import *
from .impl import *
from .util import *


def extract(
    model,
    X,
    y,
    trainSize=0.7,
    outputPath=None,
    isClassify=True,
    featureNames=[],
    catFeaturesInds=[],
    numFeaturesInds=[],
    distType="CategoricalGaussianMixture",
    tgtScore=None,
    minGain=None,
    maxSize=31,
    nPts=2000,
    nPtTries=100,
    nTestPts=2000,
    nComponents=100,
    greedyCompare=False,
):
    """ Given model type and training dataset, trains models and extracts explainable Decision Tree"""
    if outputPath:
        setCurOutput(outputPath)

    log("Splitting into training {:.2f} and test {:.2f}...".format(trainSize, 1 - trainSize), INFO)
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, train_size=trainSize)
    log("Done!", INFO)

    # Step 2: Train black-box model with loaded dataset
    rf = model
    if not rf:
        raise Exception("Invalid model: " + rf)

    rfScoreFunc = f1Vec if isClassify else mseVec
    # rfTrainScore = rfScoreFunc(rf.predict, XTrain, yTrain)
    rfTestScore = rfScoreFunc(rf.predict, XTest, yTest)
    # log('Blackbox Training score: ' + str(rfTrainScore), INFO)
    log("Blackbox Test score: " + str(rfTestScore), INFO)

    # Step 3: Set up decision tree extraction inputs
    paramsLearn = ParamsLearn(tgtScore, minGain, maxSize)
    paramsSimp = ParamsSimp(nPts, nTestPts, isClassify)

    # Step 4: Setup predictive function for given model
    rfFunc = getRfFunc(rf)

    # Step 4: Distribution
    if distType == "CategoricalGaussianMixture":
        dist = CategoricalGaussianMixtureDist(
            XTrain, catFeaturesInds, numFeaturesInds, nComponents
        )
    else:
        raise Exception("Invalid distType: " + distType)

    # Step 5: Extract decision tree
    dtExtract = learnDTSimp(
        genAxisAligned, rfFunc, dist, paramsLearn, paramsSimp, featureNames=featureNames
    )

    log("Decision tree:", INFO)
    log(str(dtExtract), INFO)
    log("Node count: " + str(dtExtract.nNodes()), INFO)

    scoreFunc = f1 if isClassify else mse

    # dtExtractRelTrainScore = scoreFunc(dtExtract.eval, XTrain, rf.predict(XTrain))
    dtExtractRelTestScore = scoreFunc(dtExtract.eval, XTest, rf.predict(XTest))
    # log('DT Relative training score (fidelity): ' + str(dtExtractRelTrainScore), INFO)
    log("DT Relative test score (fidelity): " + str(dtExtractRelTestScore), INFO)

    # dtExtractTrainScore = scoreFunc(dtExtract.eval, XTrain, yTrain)
    dtExtractTestScore = scoreFunc(dtExtract.eval, XTest, yTest)
    # log('DT Training score (f1/mse): ' + str(dtExtractTrainScore), INFO)
    log("DT Test score (f1/mse): " + str(dtExtractTestScore), INFO)

    if greedyCompare:
        # Step 6: Train a (greedy) decision tree
        log("Training greedy decision tree", INFO)
        maxLeaves = int((maxSize + 1) / 2)
        dtConstructor = DecisionTreeClassifier if isClassify else DecisionTreeRegressor
        dtTrain = dtConstructor(max_leaf_nodes=maxLeaves)
        dtTrain.fit(XTrain, rfFunc(XTrain))
        log("Done!", INFO)
        log("Node count: " + str(dtTrain.tree_.node_count), INFO)

        # dtTrainRelTrainScore = scoreFunc(lambda x: dtTrain.predict(x.reshape(1, -1)), XTrain, rf.predict(XTrain))
        dtTrainRelTestScore = scoreFunc(
            lambda x: dtTrain.predict(x.reshape(1, -1)), XTest, rf.predict(XTest)
        )

        # log('Greedy DT Relative training score (fidelity): ' + str(dtTrainRelTrainScore), INFO)
        log(
            "Greedy DT Relative test score (fidelity): " + str(dtTrainRelTestScore),
            INFO,
        )

        # dtTrainTrainScore = scoreFunc(lambda x: dtTrain.predict(x.reshape(1, -1)), XTrain, yTrain)
        dtTrainTestScore = scoreFunc(
            lambda x: dtTrain.predict(x.reshape(1, -1)), XTest, yTest
        )

        # log('Greedy DT Training score (f1/mse): ' + str(dtTrainTrainScore), INFO)
        log("Greedy DT Test score (f1/mse): " + str(dtTrainTestScore), INFO)

    return dtExtract
