/**
 * @remark The code looks really good. However, this file is not used in the implementation
 *         and can therefore be removed.
 * 
 *         If you decide to keep this file, consider removing the code segment at the bottom.
 */

/**************************************************
 * @brief Neural network implementation
 ***********************************************/

#pragma once
#include <vector>
#include <iostream>
#include "dense_layer.h"
#include "act_func.h"

/***************************************************
 * @brief Class implementation of a neural network
 *****************************************************/
namespace ml 
{

class NeuralNetwork
{
    public:
        /*****************************************************************************************
         * @brief creates a neural network.
         * @param inputCount            the number of inputs in the network
         * @param hiddenCount           The number of hidden nodes in the network
         * @param outputCount           The number of outputs in the network
         * @param actFunctionHiden      Activation funktion of the hidden lauer (default Relu)
         * @param actFunctionOutput      Activation funktion of the hidden lauer (default Relu)
         ********************************************************************************************/
        NeuralNetwork(const std::size_t inputCount,
                      const std::size_t hiddenCount,
                      const std::size_t outputCount,
                      const ActFunc actfuncHidden = ActFunc::Relu,
                      const ActFunc actfuncOutput = ActFunc::Relu);


        /*****************************************************************************************
         * @brief Provides the number of input of the network
         *
         * @return the number of inputs of the network 
         ********************************************************************************************/
         std::size_t inputCount() const;

       
        /*****************************************************************************************
         * @brief Provides the number of outputs of the network
         *
         * @return the number of outputs of the network 
         ********************************************************************************************/
        std::size_t outputCount() const;

        /*****************************************************************************************
         * @brief Provides the number of  stored trainingests
         *
         * @return the number of  stored trainingests
         ********************************************************************************************/
        std::size_t traingSetCount() const;


        /*****************************************************************************************
         * @brief trains the neural network with given params.
         * 
         *@param epochCount            The numbers of epocs for which to persom traning
         * @param learningRate          The learning rate used for optimization
         * 
         ********************************************************************************************/

        bool train(const std::size_t epochCount, const double learningRate = 0.01);

        /*****************************************************************************************
         * @brief perform prediction with a given input
         * 
         * @param input            Refereance to vector holding the input for which to predict
         * @return                 reference to vector holding the predited data. 
         ********************************************************************************************/

        const std::vector<double>& predict(const std::vector<double>&input);

        /*****************************************************************************************
         * @brief Ads trainingData
         * 
         * @param inout            Refereance to vector holding the input for the input sets(uppsätningar (0,0,0))
         * @param output           Output to vector holding the output sets (uppsättningar (0,0,0))
         *
         * @return                True if at leeast one training set was added, otherwise false.  
         ********************************************************************************************/
        
        bool addTraningData(const std::vector<std::vector<double>>& input,
                            const std::vector<std::vector<double>>& output);

        /*****************************************************************************************
         * @brief Prints training result
         *@param ostream        reference to output stream (default trerminal print)
         ********************************************************************************************/
        void printResults(std::ostream& ostream = std::cout);

    private:

    /*****************************************************************************************
     * @brief Perform feedforward operation
     * 
     *@param input Reference to input used for the operation.
    ********************************************************************************************/
    void feedforward(const std::vector<double>& input);

    /*****************************************************************************************
     * @brief Perform backpropagation operation
     * 
     *@param output Reference to output containing referce vallue
    ********************************************************************************************/
   void backpropagation(const std::vector<double>& output);

       /*****************************************************************************************
     * @brief Perform optimization 
     * 
     *@param input Reference to input used for the operation.
     *@param learningRate leraningrate to use for the optimization.
    ********************************************************************************************/
   void optimization(const std::vector<double>& output, const double learningRate);

    dense_layer myHiddenLayer;
    dense_layer myOutputLayer;
    std::vector<std::vector<double>>myTrainingInput;
    std::vector<std::vector<double>>myTrainingOutput; 


};

} //end namespace ml

/** @remark Consider removing the example code segment below. */
#if 0
NeuralNetwork network{2, 3, 1, ActFunc::Tanh } // två ingångar, 3 dolda noder och 1 utgång

const std::vector<std::vector<double>> input{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
const std::vector<std::vector<double>> output{{0}, {1}, {1}, {0}};

NeuralNetwork network{2, 3, 1, ActFunc::Tanh } // två ingångar, 3 dolda noder och 1 utgång
network.train(1000U, 0.01);

 const auto predict(network.predict(input[0]));

 network.printResults();

std::ofstream file("text.txt");
network.printResults(file);

#endif