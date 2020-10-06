## SOM: Self Organizing Map C/C++ library
It is a full stack implementation of this powerful Neural Network-based methods for unsupervised grouping of data samples.
The motivation for this implementation came around 2014-2015 since there were no available `fast` and `flexible` implementation of this nice method in C/C++.
I made it relatively simple and scalable with a possibiliy to adjust many parameters. 


## Project files
* include/SelfOrganizingMaps.h - library class definition
* src/SelfOrganizingMaps.cpp - functions implementation
* tests/test_SelfOrganizingMaps.cpp - extensive demonstration of functionality and relevant examples
* iris.txt - test data


## Examples of usage
Tests file include following functionality:
1. Reading input training data
2. Covariance matrix calculation
3. Solution of improper Inversed Covariance matrix - Tikhonov regularization
4. Pearson Correlation
5. Eigen decomposition
6. Extracting 1st and 2nd eigenvalues by ordering values
7. Calculation of optimal SOM size (height,width): "rule of thumb"; Vessanto method and Shalaginov method (check reference in the end of this document)
8. Self Organizing Map training and Best Maching Unit (BMU) calculation
9. Output of the trained groups per SOM node


## Library API

```c
//SOM Object initialization
neuralnetworks::SelfOrganizingMaps obj(numFeatures, height, width);

//Pusing training data to the SOM class object
obj.trainingData.swap(trainingData);

//Weight initialization
obj.weightsInitialization(0.1, 0.5);

//SOM training
obj.somTraining(size, 0.1);

// Trained model resuls in obj.assignedNode(i, j) objects of SOM node (i,j):
std::map<unsigned int, unsigned int >::iterator it;
for (unsigned int i = 0; i < height; i++) {
    for (unsigned int j = 0; j < width; j++) {
        printf("SOM node (%d,%d). IDs of elements: ", i, j);
        for (it = obj.assignedNode(i, j).begin(); it != obj.assignedNode(i, j).end(); it++) {
            printf("%d ", it->first);
        }
        printf("\n");
    }
}
/** API definition of the contained with trained data - list of input data IDs per SOM node
* 3d matrix that corresponds to lattice of the training data that assigned to specific nodes in SOM. \n
* Indexes: 1st - height, 2nd - width, 3rd - map of IDs of assigned training data, ordered in std::map to avoid repetitions
*/
boost::numeric::ublas::matrix<std::map<unsigned int, unsigned int > > assignedNode;
     
```


## Output example on Iris dataset
```bash
Tue Oct  6 15:15:10 2020
1601990110 seconds since the start of the Age :)
OMP threads 6 
Eigen threads 6
Program threads 6
iris.txt
Training Data Container Size: 150
Max Training Data Container Capacity: 256
Max Size of Training Data Container: 384307168202282325

Covariance Matrix:
  0.681122 -0.0390067    1.26519   0.513458
-0.0390067   0.186751  -0.319568  -0.117195
   1.26519  -0.319568    3.09242    1.28774
  0.513458  -0.117195    1.28774   0.578532

Pearson Matrix:
        1 -0.109369  0.871753  0.817952
-0.109369         1 -0.420515 -0.356543
 0.871753 -0.420515         1  0.962756
 0.817952 -0.356543  0.962756         1

Eigenvalues Matrix:
  2.91082         0         0         0
        0  0.147355         0         0
        0         0  0.921221         0
        0         0         0 0.0206086

E0 = 2.910815 E1= 0.921221 
 
Proposed : 25
Avg Pearson correlation: 0.589815
Rule of thumb : 61.2372
Vesanto : 193.493
Vesanto lower : 48.3734
Vesanto upper : 773.974
SOM final weight : 5
SOM final height : 5
SOM Object initialization...done

Train data written to SOM object
SOM Weights initialization...done
SOM Training...done
SOM node (0,0). IDs of elements: 76 77 106 111 113 115 120 121 122 124 127 128 129 133 134 135 138 140 141 143 144 145 149 
SOM node (0,1). IDs of elements: 13 32 40 46 50 51 56 61 65 67 75 77 78 86 88 90 91 102 104 105 107 108 110 112 120 134 136 137 138 139 144 147 
SOM node (0,2). IDs of elements: 61 
SOM node (0,3). IDs of elements: 
SOM node (0,4). IDs of elements: 
SOM node (1,0). IDs of elements: 70 113 127 146 
SOM node (1,1). IDs of elements: 78 85 91 97 138 
SOM node (1,2). IDs of elements: 92 93 
SOM node (1,3). IDs of elements: 30 49 
SOM node (1,4). IDs of elements: 18 
SOM node (2,0). IDs of elements: 55 68 78 81 84 
SOM node (2,1). IDs of elements: 88 96 99 
SOM node (2,2). IDs of elements: 
SOM node (2,3). IDs of elements: 45 
SOM node (2,4). IDs of elements: 0 1 3 5 7 11 12 14 16 19 22 24 25 30 31 32 33 34 44 46 47 49 
SOM node (3,0). IDs of elements: 53 59 80 92 106 
SOM node (3,1). IDs of elements: 64 
SOM node (3,2). IDs of elements: 
SOM node (3,3). IDs of elements: 
SOM node (3,4). IDs of elements: 1 2 3 6 8 12 13 22 34 42 45 
SOM node (4,0). IDs of elements: 
SOM node (4,1). IDs of elements: 60 
SOM node (4,2). IDs of elements: 
SOM node (4,3). IDs of elements: 
SOM node (4,4). IDs of elements: 41 
```


## Building and test (with working example)
This is a Debug project from NetBeans that can be built and run straignt-away on x86-64 architecture.
```bash
git clone https://github.com/ashalaginov/SOM-Self-Organizing-Map-C-library
cd SOM-Self-Organizing-Map-C-library
make 
make test
```
OR library / test compilation (to adjust path)
```bash
git clone https://github.com/ashalaginov/SOM-Self-Organizing-Map-C-library
cd SOM-Self-Organizing-Map-C-library
# Library compilation
g++ -m64 -std=c++11 -fopenmp -O0 -DEIGEN_NO_DEBUG   -c -g -Iinclude -I/usr/include/eigen3 -std=c++11 -fPIC  -o build/Debug/GNU-Linux/src/SelfOrganizingMaps.o src/SelfOrganizingMaps.cpp
g++ -m64 -std=c++11 -fopenmp -O0 -DEIGEN_NO_DEBUG    -o dist/Debug/GNU-Linux/libSOM-Self-Organizing-Map-C-library.so build/Debug/GNU-Linux/src/SelfOrganizingMaps.o -L/usr/include/boost -lpthread -shared -fPIC
# Tests compilation
g++ -m64 -std=c++11 -fopenmp -O0 -DEIGEN_NO_DEBUG   -c -g -Iinclude -I/usr/include/eigen3 -I. -std=c++11 -o build/Debug/GNU-Linux/tests/tests/test_SelfOrganizingMaps.o tests/test_SelfOrganizingMaps.cpp
g++ -m64 -std=c++11 -fopenmp -O0 -DEIGEN_NO_DEBUG    -o build/Debug/GNU-Linux/tests/TestFiles/f1 build/Debug/GNU-Linux/tests/tests/test_SelfOrganizingMaps.o build/Debug/GNU-Linux/src/SelfOrganizingMaps_nomain.o -L/usr/include/boost   
```


## Software requirements
For the Linux/Debian (kernel version 5.4+) environment there were used following packages for the core functionality:
* g++-9 - 9.3.0-10ubuntu2: GNU C++ compiler
* libgomp1:amd64 - 10-20200411-0ubuntu1: GCC OpenMP (GOMP) support library
* libeigen3-dev - 3.3.7-2: lightweight C++ template library for linear algebra
* libboost1.71-all-dev - 1.71.0-6ubuntu6: Boost C++ Libraries development files (ALL)

However, having all dependencies met, the project can be most likely compiled using g++ v4.6 / 4.8 too.


## Dataset
For the sake of demonstration, I used foamous iris dataset in the format [att1 att2 att3 att4 class] with numeric values stored in txt.


## Original Paper
You can find more information about the method, performance, parameters and practical examples in the following literatures:

	@phdthesis{shalaginov2018advancing,
            title={Advancing Neuro-Fuzzy Algorithm for Automated Classification in Largescale Forensic and Cybercrime Investigations: Adaptive Machine Learning for Big Data Forensic},
            author={Shalaginov, Andrii},
            year={2018},
            school={Norwegian University of Science and Technology}
        }

Or using following link: [FULL TEXT](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2491724)


## Misc
Flag `-O0` is used to ensure that there is no optimization is used during compilation process. Otherwise, it will break mamory allocation, e.g. Eigen decomposition vectors. 
Also `-fPIC` will ensure compilation of Position Independent Code to ensure smooth operations cross-plaftorm.