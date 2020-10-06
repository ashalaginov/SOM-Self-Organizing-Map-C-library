/* 
 * \file:   test_SelfOrganizingMaps.cpp
 * \author: Andrey Shalaginov
 * Created on Oct 29, 2014, 11:07:19 AM
 * \brief Implementation of NF based on the SOM class. For methods-specific details check:
 *  Andrii Shalaginov, "Advancing Neuro-Fuzzy Algorithm for Automated Classification in Largescale Forensic and Cybercrime Investigations: 
 *  Adaptive Machine Learning for Big Data Forensic", PhD thesis, Norwegian University of Science and Technology, 2018
 * \changelog
 * 1.0.1 (14.09.2015) - added multi-class support
 */


#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <ctime> //performace measurements
#include <cstdlib>


/**
 * Include SOM library
 * 
 */
#include<SelfOrganizingMaps.h>

//Eigen containers
#include<Eigen/Core>
#include<Eigen/Dense>
#include<Eigen/Eigenvalues>


//General
#define numClasses 3 //ID of classes are defined in the file as integers: 1, 2, 3. Has to start from 1. Add: intelligent detection of number of classes
#define numFeatures 4
#define errorThreshold 1e-6
#define maxThreads 6
#define TikhonovCorrection 1e-6

/*
 * Simple C++ Test Suite
 */
void test1() {
    std::cout << "test_SelfOrganizingMaps test 1" << std::endl;

    try {
        std::time_t result = std::time(nullptr);
        std::cout << std::asctime(std::localtime(&result))
                << result << " seconds since the start of the Age :)\n";

        omp_set_dynamic(1);
        omp_set_num_threads(maxThreads);
        Eigen::initParallel();
        Eigen::setNbThreads(maxThreads);
        printf("OMP threads %d \n", omp_get_max_threads());
        printf("Eigen threads %d\n", Eigen::nbThreads());
        printf("Program threads %d\n", maxThreads);

        boost::numeric::ublas::vector<double> inputDataAttributes(numFeatures);
        std::vector<int> classIDTrain; //train and test classes
        unsigned int height = 3, width = 3; //initial set of parameters for SOM
        //Read and push tain data
        FILE *pFileTrain; //pointer to train and test files
        //unsigned int cl;
        double cl;
        unsigned int size = 0; //temporary variable
        double tmp1; //temporary variables
        srand(time(NULL)); //initialize random seed:
        std::vector<boost::numeric::ublas::vector<double> > trainingData;

        //---------------- READING TRAIN DATA------------------------------
        unsigned int fileStatus;
        char* homedir = getenv("HOME");
        std::string path = homedir;
        path = "iris.txt"; //4 attributes + (3 classes)
        if ((pFileTrain = fopen(path.c_str(), "rt")) == NULL)
            puts("Error while opening input train file!");
        std::cout << path.c_str() << std::endl;

        //Parsing train file content into data structure
        while (!feof(pFileTrain)) {
            //sample_type sample;
            for (int i = 0; i < numFeatures; i++) {
                fileStatus = fscanf(pFileTrain, "%lf ", &tmp1);
                inputDataAttributes(i) = tmp1;
                //printf("%lf ", tmp1);
            }

            fileStatus = fscanf(pFileTrain, "%lf ", &cl);
            //printf("cl %fl \n", floor(cl));
            classIDTrain.push_back(floor(cl));
            fileStatus = fscanf(pFileTrain, "\n");
            trainingData.push_back(inputDataAttributes);
            size++;
        }
        fclose(pFileTrain);

        std::cout << "Training Data Container Size: " << trainingData.size() << "\n";
        std::cout << "Max Training Data Container Capacity: " << trainingData.capacity() << "\n";
        std::cout << "Max Size of Training Data Container: " << trainingData.max_size() << "\n";
        //---------------- END OF READING TRAIN DATA------------------------------

        //---------------- ANALYTICS OF SOM SIZE------------------------------
        double Smin = 2 * 2, Smax = 5 * 5, alpha = 0.0, Sproposed = 0, Sv = 0, Svlower = 0, Svupper = 0, Sthumbrule = 0;

        Eigen::MatrixXd tmpObservations;
        Eigen::MatrixXd centered;
        Eigen::MatrixXd covariance;
        Eigen::MatrixXd correlation;
        //Observation to Eigen matrix type

        tmpObservations.resize(size, numFeatures);
        correlation.resize(numFeatures, numFeatures);

        //Eigen::VectorXd tmpVector;
        for (unsigned int i = 0; i < size; i++)
            tmpObservations.row(i) = Eigen::VectorXd::Map(&trainingData[i][0], trainingData[i].size()).transpose();
        //!!!!MIGHT CAUSE DAMAGE since the vectors might not be in contigency memory!

        //Covariance matrix 
        centered = tmpObservations.rowwise() - tmpObservations.colwise().mean();
        covariance = (centered.adjoint() * centered) / double(tmpObservations.rows());
        //Temporarily disabled due to long output
        std::cout << "\nCovariance Matrix:\n" << covariance << std::endl;

        //Solution of improper Inversed Covariance matrix - Tikhonov correction
        Eigen::VectorXd diag(numFeatures);
        Eigen::MatrixXd diagonal(numFeatures, numFeatures);
        for (unsigned int i = 0; i < numFeatures; i++)
            diag(i) = TikhonovCorrection;
        diagonal = diag.asDiagonal();
        covariance = covariance + diagonal;

        //Pearson Correlation
        for (unsigned int i = 0; i < covariance.rows(); i++)
            for (unsigned int j = 0; j < covariance.cols(); j++)
                correlation(i, j) = covariance(i, j) / sqrt(covariance(i, i) * covariance(j, j));
        //Temporarily disabled due to long output
        std::cout << "\nPearson Matrix:\n" << correlation << std::endl;

        //Eigen decomposition
        Eigen::EigenSolver<Eigen::MatrixXd> es(correlation);
        Eigen::MatrixXd D = es.pseudoEigenvalueMatrix();
        //Temporarily disabled due to long output
        std::cout << "\nEigenvalues Matrix:\n" << D << std::endl;

        //Extract 1st and 2nd eigenvalues by ordering values
        std::map<double, double> mapTmp;
        std::map<double, double>::reverse_iterator rit;
        unsigned int m = 0;
        double E[2];
        for (m = 0; m < numFeatures; m++)
            mapTmp.insert(std::pair<double, double>(D(m, m), D(m, m)));

        m = 0;
        for (rit = mapTmp.rbegin(); rit != mapTmp.rend() && m < 2; ++rit) {
            E[m] = rit->first;
            m++;
        }

        printf("\nE0 = %f E1= %f \n ", E[0], E[1]);

        //Alpha calculation
        alpha = E[0] / E[1] * (correlation.array().abs().sum() - (double) numFeatures) / (double) (numFeatures * numFeatures - numFeatures) * numClasses;
        if (alpha > 1)
            alpha = 1;

        //Proposed optimal size of SOM
        Sproposed = Smin + (Smax - Smin) * alpha;
        std::cout << "\n Proposed :\n" << Sproposed << std::endl;
        std::cout << "\n Avg Pearson :\n" << (correlation.array().abs().sum() - (double) numFeatures) / (double) (numFeatures * numFeatures - numFeatures) << std::endl;

        Sv = 5 * sqrt(size) * E[0] / E[1];
        Svlower = 0.25 * Sv;
        Svupper = 4 * Sv;

        Sthumbrule = 5 * sqrt(size);

        std::cout << "\n Rule of thumb :\n" << Sthumbrule << std::endl;
        std::cout << "\n Vesanto :\n" << Sv << std::endl;
        std::cout << "\n Vesanto lower :\n" << Svlower << std::endl;
        std::cout << "\n Vesanto upper :\n" << Svupper << std::endl;

        double S = Sproposed; //SELECTION OF SOM SIZE CALCULATION METHOD: Sv - Vesanto, Sproposed - proposed methods, Sthumbrule - "rule of thumb". Check references

        height = ceil(sqrt(S));
        width = floor(S / height);
        std::cout << "\n weight :\n" << width << std::endl;
        std::cout << "\n height :\n" << height << std::endl;

        //free
        tmpObservations.resize(0, 0);
        centered.resize(0, 0);
        covariance.resize(0, 0);
        correlation.resize(0, 0);
        diag.resize(0);
        diagonal.resize(0, 0);
        D.resize(0, 0);
        mapTmp.clear();
        //---------------- END OF ANALYTICS OF SOM SIZE------------------------------

        //----------------SOM TRAINING------------------------------
        //Initialize SOM object class
        neuralnetworks::SelfOrganizingMaps obj(numFeatures, height, width);
        puts("\nSOM Object initialization...done\n");

        //Write to the SOM class object
        obj.trainingData.swap(trainingData);
        trainingData.clear();
        puts("Train data written to SOM object");

        //Weight Initialization
        obj.weightsInitialization(0.1, 0.5);
        puts("SOM Weights initialization...done");

        //SOM TRAINING
        //Use size/100 for bootstrap
        obj.somTraining(size, 0.1);
        puts("SOM Training...done");

        FILE *pSOMclustering; //pointer to file with SOM clusters statistics
        if ((pSOMclustering = fopen("SOM_clustering.csv", "wt")) == NULL)
            puts("Error while opening input train file!");
        fprintf(pSOMclustering, "SOM_hight SOM_width Class Samples\n");
        //----------------END OF SOM TRAINING------------------------------


        //----------------SOM RESULTS - ------------------------------
        //Just for debug purpose - clusters distribution
        std::map<unsigned int, unsigned int >::iterator it;
        for (unsigned int i = 0; i < height; i++)
            for (unsigned int j = 0; j < width; j++) {

                double mean = 0, mean0 = 0, mean1 = 0;
                unsigned int cl0 = 0, cl1 = 0;
                printf("SOM node (%d,%d). IDs of elements: ", i, j);
                for (it = obj.assignedNode(i, j).begin(); it != obj.assignedNode(i, j).end(); it++) {

                    mean += obj.trainingData[it->first](0);
                    printf("%d ", it->first);

                    if (classIDTrain[it->first] == 1) {
                        mean0 += obj.trainingData[it->first](0);
                        cl0++;
                    }
                    if (classIDTrain[it->first] == 2) {
                        mean1 += obj.trainingData[it->first](0);
                        cl1++;
                    }
                }
                printf("\n");
                if (cl0 > 0)fprintf(pSOMclustering, "%d %d 1 %d\n", i, j, cl0);
                if (cl1 > 0)fprintf(pSOMclustering, "%d %d 2 %d\n", i, j, cl1);
            }
        fclose(pSOMclustering);

        //Free the memory
    } catch (std::runtime_error e) {

        std::cout << e.what();
    }
}

int main(int argc, char** argv) {
    std::cout << "\n%SUITE_STARTING% test_SelfOrganizingMaps\n" << std::endl;
    std::cout << "\n%SUITE_STARTED%\n" << std::endl;

    std::cout << "\n%TEST_STARTED% test1 (test_SelfOrganizingMaps)\n" << std::endl;
    test1();
    /*
    std::cout << "%TEST_FINISHED% time=0 test1 (test_SelfOrganizingMaps)" << std::endl;

    std::cout << "%TEST_STARTED% test2 (test_SelfOrganizingMaps)\n" << std::endl;
    test2();
    std::cout << "%TEST_FINISHED% time=0 test2 (test_SelfOrganizingMaps)" << std::endl;

    std::cout << "%SUITE_FINISHED% time=0" << std::endl;
     */
    //getchar();
    return (EXIT_SUCCESS);
}

