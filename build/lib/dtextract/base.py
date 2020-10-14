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

import numpy as np

from .data import *
from .util import *
from .impl import *


# This function
# (1) Loads the dataset
# (2) Trains a given model
# (3) Extracts a decision tree from the trained random forest
#
# parameters/returns:
#  datasetPath : str (path to csv containing dataset)
#  hasHeader : bool (whether the csv has a header)
#  dataTypes : [int] (labels for dataset column data types)
#  isClassify : bool (classification vs. regression)
#  distType: 'CategoricalGaussianMixture' , the type of distribution
#  return : float * float * float (the relative accuracy according to the distribution for the extracted dt, trained dt, and lasso)


def extract(model,
            datasetPath,
            dataTypes,
            testTrainSplit=0.7,
            isClassify=True,
            hasHeaders=None,
            headers=None,
            distType="CategoricalGaussianMixture",
            tgtScore=None,
            minGain=None,
            maxSize=31,
            nPts=2000,
            nPtTries=100,
            nTestPts=2000,
            nComponents=100):
    """ Given model type and training dataset, trains models and extracts explainable Decision Tree"""
    # Step 1: Load training dataset
    log('Parsing CSV...', INFO)
    (df, res, resMap, catFeats) = readCsv(datasetPath, hasHeaders, dataTypes, headers)
    featureNames = list(df.columns)
    log('Done!', INFO)

    log('Splitting into training and test...', INFO)
    (trainDf, testDf) = split(df, testTrainSplit)
    log('Done!', INFO)

    log('Constructing data matrices...', INFO)
    (XTrain, yTrain, catFeatIndsTrain, numericFeatIndsTrain) = constructDataMatrix(trainDf, res, catFeats)
    (XTest, yTest, catFeatIndsTest, numericFeatIndsTest) = constructDataMatrix(testDf, res, catFeats)
    log('Done!', INFO)

    # Step 2: Train black-box model with loaded dataset
    log('Training model: {}...'.format(model.__class__), INFO)
    rf = model()
    rf.fit(XTrain, yTrain)
    log('Done!', INFO)

    rfScoreFunc = f1Vec if isClassify else mseVec

    rfTrainScore = rfScoreFunc(rf.predict, XTrain, yTrain)
    rfTestScore = rfScoreFunc(rf.predict, XTest, yTest)

    log('Training score: ' + str(rfTrainScore), INFO)
    log('Test score: ' + str(rfTestScore), INFO)

    # Step 3: Set up decision tree extraction inputs
    paramsLearn = ParamsLearn(tgtScore, minGain, maxSize)
    paramsSimp = ParamsSimp(nPts, nTestPts, isClassify)

    # Step 4: Setup predictive function for given model
    rfFunc = getRfFunc(rf)

    # Step 4: Distribution
    if distType == 'CategoricalGaussianMixture':
        dist = CategoricalGaussianMixtureDist(XTrain, catFeatIndsTrain, numericFeatIndsTrain, nComponents)
    else:
        raise Exception('Invalid distType: ' + distType)

    # Step 5: Extract decision tree
    dtExtract = learnDTSimp(genAxisAligned, rfFunc, dist, paramsLearn, paramsSimp, featureNames=featureNames)

    log('Decision tree:', INFO)
    log(str(dtExtract), INFO)
    log('Node count: ' + str(dtExtract.nNodes()), INFO)

    return dtExtract
