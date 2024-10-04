/*******************************************************************************
 * @brief Demonstration of a dense layer network.
 ******************************************************************************/
#include <iostream>
#include <vector>

#include "dense_layer.h"
#include "utils.h"

using namespace ml;

/*******************************************************************************
 * @brief Creates and initializes a dense layer network, which is trained to 
 *        detect a 2-bit XOR pattern. 
 *  
 *        The dense layer network consists of:
 * 
 *        - A hidden layer with three nodes, using the hyperbolic tangent (tanh) 
 *         as activation function.
 * 
 *        - An output layer with one node, using the ReLU (Rectified Linear Unit) 
 *          as activation function.
 * 
 *        Practically, the dense layer network forms a small neural network
 *        consisting of two inputs, three hidden nodes and one output.
 *  
 *        The network is trained during 1000 epochs with a 10 % learning rate.
 *        The results post training are printed in the terminal.      
 * 
 * @return Success code 0 upon termination of the program.
 ******************************************************************************/
int main()
{
    // Defines network parameters and training sets.
    constexpr std::size_t epochCount{1000U};
    constexpr double learningRate{0.1};
    const std::vector<std::vector<double>> inputSets{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const std::vector<std::vector<double>> referenceSets{{0}, {1}, {1}, {0}};

    // Creates the dense layers.
    dense_layer hiddenLayer{3U /*  noder */, 2U /*vikt*/, act_func::tanh};
    dense_layer outputLayer{1U, 3U};
    

    // Trains the dense layer network.
    for (std::size_t i{}; i < epochCount; ++i)
    {
        for (std::size_t j{}; j < inputSets.size(); ++j)
        {
            // Performs feedforward.
            hiddenLayer.feedforward(inputSets[j]);
            outputLayer.feedforward(hiddenLayer.output());

            // Performs backpropagation.
            outputLayer.backpropagate(referenceSets[j]);
            hiddenLayer.backpropagate(outputLayer);

            // Performs optimization.
            hiddenLayer.optimize(inputSets[j], learningRate);
            outputLayer.optimize(hiddenLayer.output(), learningRate);
        }
    }

    // Prints the results in the terminal.
    for (const auto& input : inputSets)
    {
        std::cout << "Input: ";
        utils::vector::print(input, std::cout, ", ");

        hiddenLayer.feedforward(input);
        outputLayer.feedforward(hiddenLayer.output());

        std::cout << "output: ";
        utils::vector::print(outputLayer.output());
    }
    return 0;
}