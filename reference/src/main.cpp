/*******************************************************************************
 * @brief Demonstration of a dense layer network.
 ******************************************************************************/
#include <iostream>
#include <vector>

#include "dense_layer.h"
#include "utils.h"
#include "neural_network.h"

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
    const std::vector<std::vector<double>> input_sets{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const std::vector<std::vector<double>> reference_sets{{0}, {1}, {1}, {0}};

    // Creates the dense layers.
    dense_layer hidden_layer{3U /*  noder */, 2U /*vikt*/, act_func::tanh};
    dense_layer output_layer{1U /* vikt */, 3U /* noder */};

    dense_layer::train(hidden_layer, output_layer, input_sets, reference_sets, 1000 /*epocher */, 0.1 /*learningRate*/);
    
    dense_layer::print(hidden_layer, output_layer, input_sets, reference_sets);
}