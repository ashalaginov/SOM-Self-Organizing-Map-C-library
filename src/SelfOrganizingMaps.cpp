/* 
 * \file   SelfOrganizingMaps.cpp
 * \brief Implementation of the Self-Organizing Maps
 * \author Andrey Shalaginov 
 * \version 1.0
 * \date October 29, 2014, 11:06 AM
 * \copyright Andrey Shalaginov
 */

/**
 * Include own header 
 */
#include<SelfOrganizingMaps.h>

using namespace neuralnetworks;

SelfOrganizingMaps::SelfOrganizingMaps(unsigned int inputDimension, unsigned int somHeight, unsigned int somWidth) : weightsLattice(somHeight, somWidth), assignedNode(somHeight, somWidth) {
    if (inputDimension == 0) {
        std::string str("Error! The dimension is 0!");
        throw std::runtime_error(str.c_str());
    }
    if (somHeight == 0) {
        std::string str("Error! The SOM heigh is 0!");
        throw std::runtime_error(str.c_str());
    }
    if (somWidth == 0) {
        std::string str("Error! The SOM width is 0!");
        throw std::runtime_error(str.c_str());
    }

    //Initialization of the weights lattice with corresponding dimension of the input data
    boost::numeric::ublas::vector<double> tmpVector(inputDimension);
    for (unsigned int i = 0; i < somHeight; i++)
        for (unsigned int j = 0; j < somWidth; j++)
            weightsLattice(i, j) = tmpVector;

    //Initialize the random generator
    srand((unsigned int) time(NULL));

    //initialize private variables
    dimension = inputDimension;
    width = somWidth;
    height = somHeight;

    //Calculate the biggest possible initial radius (he half of width either height (delta_0))
    sigma0 = (double) std::max(height, width) / 2;
}

SelfOrganizingMaps::~SelfOrganizingMaps() throw () {
    //Free memory
    std::vector<boost::numeric::ublas::vector<double> >().swap(trainingData);
    boost::numeric::ublas::matrix<boost::numeric::ublas::vector<double> > ().swap(weightsLattice);
}

void SelfOrganizingMaps::weightsInitialization(double a, double b) {
    if (a < 0 || a > 1 || b < 0 || b > 1) {
        std::string str("Error! Range a..b for rules initialization should be small (0..1)");
        throw std::runtime_error(str.c_str());
    }
    //Fill the 3d array of weight lattice with random values
    for (unsigned int i = 0; i < weightsLattice.size1(); i++)
        for (unsigned int j = 0; j < weightsLattice.size2(); j++)
            for (unsigned int k = 0; k < dimension; k++)
                //weightsLattice(i, j)(k) = (double) (b - (b - a) / 2);
                weightsLattice(i, j)(k) = a + (rand() / (RAND_MAX / (b - a)));
}

double SelfOrganizingMaps::nodeDistance(const boost::numeric::ublas::vector<double> &inputDataAttributes, unsigned int nodeHeight, unsigned int nodeWeight) {
    if (inputDataAttributes.size() != 0) {
        double tmp = 0;
        //Euclidean distance
        for (unsigned int i = 0; i < inputDataAttributes.size(); i++) {
            tmp += (double) pow(inputDataAttributes(i) - weightsLattice(nodeHeight, nodeWeight)(i), 2);
        }
        return (double) sqrt(tmp);
    } else {
        std::string str("Error! Vector of input data attributes is empty (can not calculate the distance)!");
        throw std::runtime_error(str.c_str());
    }
}

const std::vector<unsigned int> SelfOrganizingMaps::bestMatchingUnit(const boost::numeric::ublas::vector<double> &inputDataAttributes) {
    unsigned int bmuHeight = 0,
            bmuWidth = 0;
    double minDistance = DBL_MAX,
            tmp;

    //find the closest node that will be a BMU
    for (unsigned int i = 0; i < weightsLattice.size1(); i++)
        for (unsigned int j = 0; j < weightsLattice.size2(); j++) {
            tmp = nodeDistance(inputDataAttributes, i, j);
            //Check if the node is the closest than before
            if (tmp < minDistance) {
                minDistance = tmp;
                bmuHeight = i;
                bmuWidth = j;
            }
        }
    std::vector<unsigned int> tmpCoordinates;
    tmpCoordinates.push_back(bmuHeight);
    tmpCoordinates.push_back(bmuWidth);
    return tmpCoordinates;
}

double SelfOrganizingMaps::currentNeighbourhoodRadius(unsigned int currentIteration) {
    //Calculate the current neighborhood radius (sigma(t)) as a function from the time
    return sigma0 * std::exp(-(double) currentIteration / lambda);
}

double SelfOrganizingMaps::currentLearningRate(unsigned int currentIteration) {
    //Calculate the current learning rate (L_t))
    return learningRate * exp(-(double) currentIteration / lambda);
}

void SelfOrganizingMaps::weightsUpdate(const std::vector<unsigned int> & BMU, unsigned int currentIteration, const boost::numeric::ublas::vector<double>& inputDataAttributes) {
    //Current Learning Rate
    double lRate = currentLearningRate(currentIteration);

    //Current Radius (sigma)
    double radius = currentNeighbourhoodRadius(currentIteration);
    double tmpDist, theta; //calculate the effect from learning based on the distance from BMU

    //Update weights of nodes within specific distance from the BMU
    for (unsigned int i = 0; i < weightsLattice.size1(); i++)
        for (unsigned int j = 0; j < weightsLattice.size2(); j++) {
            //Euclidean Distance from BMU to current node
            tmpDist = pow((double) BMU[0] - (double) i, 2) + pow((double) BMU[1] -(double) j, 2);
            //Effect on the learning from how far the node is located from BMU (theta(t))
            theta = exp(-tmpDist / (2 * pow(radius, 2)));
            //Update weights
            for (unsigned int k = 0; k < dimension; k++)
                weightsLattice(i, j)(k) = weightsLattice(i, j)(k) + lRate * theta * (inputDataAttributes(k) - weightsLattice(i, j)(k));
        }
}

void SelfOrganizingMaps::somTraining(unsigned int epochs, double learningStep) {
    if (epochs == 0) {
        std::string str("Error! The amount of epochs should not be 0!");
        throw std::runtime_error(str.c_str());
    }
    if (learningStep > 1 || learningStep <= 0) {
        std::string str("Error! The learning step should be in the range (0,1]!");
        throw std::runtime_error(str.c_str());
    }

    //Initialize private variables
    learningRate = learningStep;
    Epochs = epochs;

    //Time constant 
    lambda = (double) Epochs / log(sigma0);

    //Update weights in neighborhood
    unsigned int j = 0;
    std::vector<unsigned int> BMUcoordinates;
    //FILE *pFileTrain; //pointer to train and test files

    //The training process


    for (unsigned int i = 0; i < Epochs; i++) {
        //Randomly select input data 
        j = rand() % trainingData.size();

        //Find BMU
        BMUcoordinates.clear();
        BMUcoordinates = bestMatchingUnit(trainingData[j]);
        assignedNode(BMUcoordinates[0], BMUcoordinates[1]).insert(std::pair<unsigned int, unsigned int >(j, j));

        //Update weights (can be cyclic application of the training samples)
        weightsUpdate(BMUcoordinates, i, trainingData[j]);
    }

   // trainingOrder.swap(tmp);
    std::vector<unsigned int> ().swap(BMUcoordinates);
}

void SelfOrganizingMaps::pushData(const boost::numeric::ublas::vector<double>& inputDataAttributes) {
    if (inputDataAttributes.size() == 0 || inputDataAttributes.size() != dimension) {
        std::string str("Error! The fed vector of attributes has a wrong dimensionality!!");
        throw std::runtime_error(str.c_str());
    }
    trainingData.push_back(inputDataAttributes);
}

const boost::numeric::ublas::matrix<boost::numeric::ublas::vector<double> >& SelfOrganizingMaps::returnWeightsLattice() {
    return weightsLattice;
}