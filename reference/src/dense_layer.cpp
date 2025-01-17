#include <iostream>
#include <vector>

#include "dense_layer.h"
#include "utils.h"

#include <exception>

namespace ml
{

/// konstruktor
dense_layer::dense_layer(const std::size_t node_count , const std::size_t weight_count,
                            const act_func act_func)
    : myOutput(node_count, 0) // Outputs are set to 0 as default.
    , myError(node_count, 0)
    , myBias(node_count, 0)
    , myWeights(node_count,std::vector<double>(weight_count, 0)) // node_count x weight_count weights.
    , myActFunc{act_func} 
{
    // fixar medlemsmariablerna som inte är satt i konstrutorn här

    utils::vector::initRandom<double>(myBias, node_count, 0, 1);
    utils::vector::initRandom<double>(myWeights, node_count, weight_count, 0, 1);

    if (node_count == 0U) { throw std::invalid_argument("Cannot create layer without nodes!\n"); }

    if (weight_count == 0U)  { throw std::invalid_argument("Cannot create layer without weights!\n"); }

    std::cout << "We have " << myOutput.size() << " nodes and ";
    std::cout << myWeights[0].size() << " weights per node!\n";


} // end of construktor

//----------getter-----------------------
std::size_t dense_layer::node_count() const { return myOutput.size(); }

std::size_t dense_layer::weight_count() const { return myWeights[0].size(); }

const std::vector<double>& dense_layer::bias() const { return myBias; }      

const std::vector<std::vector<double>>& dense_layer::weights() const { return myWeights; } 

const std::vector<double>& dense_layer::error() const { return myError; }    

const std::vector<double>& dense_layer::output() const { return myOutput; }



//----------getter-----------------------


/// --------------------Funktioner---------------

void dense_layer::feedforward(const std::vector<double>& input)
{
    if (input.size() != weight_count()) 
    {
        throw std::invalid_argument("Dimension mismatch in dense_layer::feedforward");
    }

    for (std::size_t i{}; i < node_count(); ++i) 
    {
        // sum = bias[i] + weights[i][j] * input[j] för alla vikter j (alla vikter * alla inputs).
        double sum{myBias[i]}; // Vi lägger till bias-värdet först.

        for (std::size_t j{}; j < weight_count(); ++j)
        {
            // i = nodnumret, j = viktnumret.
            sum += myWeights[i][j] * input[j]; // Därefter adderar vi vikten * input.
        }
        myOutput[i] = act_func_output(sum);
        //std::cout << "New output for node at index " << i << ": " << myOutput[i] << "\n";
    }
}   // end feedforward

void dense_layer::backpropagate(const std::vector<double>& reference)
{
    if (reference.size() != node_count()) { throw std::invalid_argument("Bakproagation reference does not match the shape of the dence layer"); }

    for (std::size_t i{}; i < node_count(); ++i)
    {
        const auto error{reference[i] - myOutput[i]};
        myError[i] = error * act_func_gradient(myOutput[i]);
    }
} // end first layer

void dense_layer::backpropagate(const dense_layer& next_layer)
{
    if (next_layer.weight_count() != node_count()) { throw std::invalid_argument("The shape of the next layer does not match the current layer!"); }

    for (std::size_t i{}; i < node_count(); ++i)
    {
        double error{};

        for (std::size_t j{}; j < next_layer.node_count(); ++j)
        {
            error += next_layer.error()[j] * next_layer.weights()[j][i];
        }
        myError[i] = error * act_func_gradient(myOutput[i]);
    }
} // end backpropagate next layer

void dense_layer::optimize(const std::vector<double>& input, const double learning_rate)
{
    if (input.size() != weight_count()) { throw std::invalid_argument("The shape of the next layer does not match the current layer!"); }

    if (learning_rate <= 0.0)  { throw std::invalid_argument("The learning rate must exceed 0!"); }    

    for (std::size_t i{}; i < node_count(); ++i)
    {
        myBias[i] += myError[i] * learning_rate;

        for (std::size_t j{}; j < weight_count(); ++j)
        {
            myWeights[i][j] += myError[i] * learning_rate * input[j];
        }
    } // end optimize
}

void dense_layer::train(dense_layer& hidden_layer, dense_layer& output_layer, 
            const std::vector<std::vector<double>>& input_sets,
            const std::vector<std::vector<double>>& reference_sets,
            std::size_t epoch_count, const double learning_rate ) // Trains the dense layer network.
    {    
        for (std::size_t i{}; i < epoch_count; ++i)
        {
            for (std::size_t j{}; j < input_sets.size(); ++j)
            {
                // Performs feedforward.
                hidden_layer.feedforward(input_sets[j]);
                output_layer.feedforward(hidden_layer.output());

                // Performs backpropagation.
                output_layer.backpropagate(reference_sets[j]);
                hidden_layer.backpropagate(output_layer);

                // Performs optimization.
                hidden_layer.optimize(input_sets[j], learning_rate);
                output_layer.optimize(hidden_layer.output(), learning_rate);
            }
        }
    }
    
void dense_layer::print(dense_layer& hidden_layer,dense_layer& output_layer, const std::vector<std::vector<double>>& input_sets,
                                         const std::vector<std::vector<double>>& reference_sets)
{
    for (const auto& input : input_sets)
    {
        std::cout << "Input: ";
        utils::vector::print(input, std::cout, ", ");

        hidden_layer.feedforward(input);
        output_layer.feedforward(hidden_layer.output());

        std::cout << "output: ";
        utils::vector::print(output_layer.output());
    }
}
// -----------------------------------------------------------------------------
double dense_layer::act_func_output(const double number) const
{
    return myActFunc == act_func::relu ? 
        utils::math::relu(number) : utils::math::tanh(number);
}

// -----------------------------------------------------------------------------
double dense_layer::act_func_gradient(const double number) const
{
    return myActFunc == act_func::relu ?
        utils::math::reluGradient(number) : utils::math::tanhGradient(number);
}
/// --------------------END Functions---------------

} // end namespace
