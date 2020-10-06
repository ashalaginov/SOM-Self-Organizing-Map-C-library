## SOM: Self Organizing Map C++ library
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
Solution of improper Inversed Covariance matrix - Tikhonov correction

### SOM size calulation

### SOM Training and Best Maching Unit (BMU) calculation


## Compilation


## Software requirements:


## Dataset
For the sake of demonstration, i was used foamous iris dataset in the format [att1 att2 att3 att4 class] with numeric values stored in txt.



## Original Paper
You can find more information about the method, performance, parameters and practical examples in the following literatures:

	@phdthesis{shalaginov2018advancing,
            title={Advancing Neuro-Fuzzy Algorithm for Automated Classification in Largescale Forensic and Cybercrime Investigations: Adaptive Machine Learning for Big Data Forensic},
            author={Shalaginov, Andrii},
            year={2018},
            school={Norwegian University of Science and Technology}
        }

Or using following link: [FULL TEXT](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2491724)


