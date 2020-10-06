/* 
 * \file   SelfOrganizingMaps.h
 * \brief Implementation of the Self-Organizing Maps, 
 * \brief thanks to ai-junkie.com for the ideas
 * \author Andrey Shalaginov 
 * \version 1.0
 * \date October 29, 2014, 11:06 AM
 * \copyright Andrey Shalaginov
 */

#ifndef SELFORGANIZINGMAPS_H
#define	SELFORGANIZINGMAPS_H

#include <float.h>
#include <stdio.h>
#include <utility> //pair
#include <math.h> //exp

/** 
 * Include STL
 */
#include <vector>
#include <string>
#include <algorithm> //max
#include <map> //map

/**
 * Include Boost
 */
#include<boost/numeric/ublas/matrix.hpp>
#include<boost/numeric/ublas/io.hpp>
#include<boost/numeric/ublas/vector.hpp>


namespace neuralnetworks {

    /**
     * Class definition
     */

    class SelfOrganizingMaps {
    private:
        /**
         * Number of training epochs
         */
        unsigned int Epochs;

        /**
         * Learning rate of the SOM
         */
        double learningRate;

        /**
         * Initial radius, which is equal to half of the lattice (w or h)
         */
        double sigma0;

        /**
         * Time constant (lamda)
         */
        double lambda;

        /**
         * 3d matrix that corresponds to lattice of the weights in SOM. \n
         * Indexes: 1st - height, 2nd - width, 3rd - dimension of the input data
         */
        boost::numeric::ublas::matrix<boost::numeric::ublas::vector<double> > weightsLattice;

        /**
         * Determine the Euclidean distance from the weights vector in a corresponding node's weight vector to an input data sample
         * @param inputDataAttributes
         * @param nodeHeight
         * @param nodeWeight
         * @return Euclidean distance value
         */
        double nodeDistance(const boost::numeric::ublas::vector<double> &inputDataAttributes, unsigned int nodeHeight, unsigned int nodeWeight);

        /**
         * Find a best matching unit
         * @param inputDataAttributes vector of input data sample attributes
         * @return std::vector<unsigned int> nodeHeight, nodeWidth
         */
        const std::vector<unsigned int> bestMatchingUnit(const boost::numeric::ublas::vector<double> &inputDataAttributes);

        /**
         * Calculate the neighborhood radius based on the current iteration
         * @param currentIteration Current training iteration
         * @return double radius
         */
        double currentNeighbourhoodRadius(unsigned int currentIteration);

        /**
         * Calculate the learning rate on the current iteration
         * @param currentIteration Current training iteration
         * @return double learning rate
         */
        double currentLearningRate(unsigned int currentIteration);

        /**
         * Update weights of all neurons in the neighborhood
         * @param BMU Coordinate of BMU for current data sample (height and width)
         * @param neighbourhoodRadius Neighborhood radius based on the current iteration
         * @param currentIteration Current training iteration
         */
        void weightsUpdate(const std::vector<unsigned int> &BMU, unsigned int currentIteration, const boost::numeric::ublas::vector<double>& inputDataAttributes);


    public:

        /**
         * Dimension of the input data samples
         */
        unsigned int dimension;

        /**
         * Weight of the SOM
         */
        unsigned int width;

        /**
         * height of the SOM
         */
        unsigned int height;

        /**
         * The vector of attribute vectors from the training data. Has to be feed into the class. \
         * Indexes: 1st - data sample id, 2nd - data sample attributes
         */
        std::vector<boost::numeric::ublas::vector<double> > trainingData;

        /**
         * 3d matrix that corresponds to lattice of the training data that assigned to specific nodes in SOM. \n
         * Indexes: 1st - height, 2nd - width, 3rd - map of IDs of assigned training data, ordered in std::map to avoid repetitions
         */
        boost::numeric::ublas::matrix<std::map<unsigned int, unsigned int > > assignedNode;

        std::vector<int> trainingOrder;


        /**
         * Constructor 
         * @param inputDimension dimension of input vector
         * @param somHeight height of the SOM lattice
         * @param somWidth width of the SOM lattice
         */
        SelfOrganizingMaps(unsigned int inputDimension, unsigned int somHeight, unsigned int somWidth);

        /**
         * Virtual Destructor 
         */
        virtual ~SelfOrganizingMaps() throw ();

        /**
         * Feeding the input data into class. Has to be done iteratively for all training data samples
         * @param inputDataAttributes The set of attributes of a single data sample.
         */
        void pushData(const boost::numeric::ublas::vector<double>& inputDataAttributes);

        /**
         * Initialization of the weights in the lattice through the random number in the range a..b (small numbers, 0..1)
         * @param a left bound
         * @param b right bound
         */
        void weightsInitialization(double a, double b);

        /**
         * Global training procedure of SOM, which includes BMU and weights update \n
         * The training is done via random selection of training data samples and equal to epochs
         * @param epochs Number of training epochs
         * @param learningStep Learning rate of the weights update procedure
         */
        void somTraining(unsigned int epochs, double learningStep);

        /**
         * Safe return of the weights lattice
         * @return boost::numeric::ublas::matrix<boost::numeric::ublas::vector<double> > 3d array
         */
        const boost::numeric::ublas::matrix<boost::numeric::ublas::vector<double> >& returnWeightsLattice();
    };

}

#endif	/* SELFORGANIZINGMAPS_H */