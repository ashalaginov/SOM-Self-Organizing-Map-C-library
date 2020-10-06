## SOM: Self Organizing Map C/C++ library
It is a full stack implementation of this powerful Neural Network-based methods for unsupervised grouping of data samples.
The motivation for this implementation came around 2014-2015 since there were no available `fast` and `flexible` implementation of this nice method in C/C++.
I made it relatively simple and scalable with a possibiliy to adjust many parameters. 

## Project files
include/SelfOrganizingMaps.h - library class definition
src/SelfOrganizingMaps.cpp - functions implementation
tests/test_SelfOrganizingMaps.cpp - extensive demonstration of functionality and relevant examples
iris.txt - test data

## Examples of usage


### Data processing
Covariance matrix calculation
Solution of improper Inversed Covariance matrix - Tikhonov regularization

### SOM size calulation

### SOM Training and Best Maching Unit (BMU) calculation


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