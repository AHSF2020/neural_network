/*******************************************************************************
 * @brief Demonstration of a dense layer network.
 ******************************************************************************/
#include <iostream>
#include <vector>
#include "dense_layer.h"
#include "button.h"
#include "led.h"
#include "utils.h"

#define ENABLE_DRIVERS

using namespace ml;
using namespace rpi;
using namespace utils;

/*******************************************************************************
 * @brief Creates and initializes a dense layer network, which is trained to 
 *        detect a 4-bit XOR pattern. 
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
    // XOR-träningsdata (4-bitars XOR-mönster)
    const std::vector<std::vector<double>> input_sets{
        {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 0, 1, 1},
        {0, 1, 0, 0}, {0, 1, 0, 1}, {0, 1, 1, 0}, {0, 1, 1, 1},
        {1, 0, 0, 0}, {1, 0, 0, 1}, {1, 0, 1, 0}, {1, 0, 1, 1},
        {1, 1, 0, 0}, {1, 1, 0, 1}, {1, 1, 1, 0}, {1, 1, 1, 1}
    };

    // Referenssätt för XOR (om summan är udda, sätt 1, annars 0)
    const std::vector<std::vector<double>> reference_sets{
        {0}, {1}, {1}, {0},
        {1}, {0}, {0}, {1},
        {1}, {0}, {0}, {1},
        {0}, {1}, {1}, {0}
    };

    // Creates the dense layers.
    dense_layer hidden_layer{4U /*  noder */, 4U /*vikt*/, act_func::tanh};
    dense_layer output_layer{1U /* vikt */, 4U /* noder */, act_func::relu};

    dense_layer::train(hidden_layer, output_layer, input_sets, reference_sets, 100000 /*epocher */, 0.01 /*learningRate*/);
    
    dense_layer::print(hidden_layer, output_layer, input_sets, reference_sets);

    #ifdef ENABLE_DRIVERS

    Button button1(23), button2(17), button3(27), button4(22);

        // Definiera GPIO-pinnar (ändra vid behov)
        std::vector<Button*> buttons = {
            &button1, &button2, &button3, &button4  // 4 tryckknappar för G-nivå
        };

    while(true) {

    std::vector<double> input(buttons.size(), 0.0);
        for(std::size_t i {}; i < buttons.size(); i++) {

            if(buttons[i]->isPressed()) {
               input[i] = 1;
            }
        }

        std::cout << "Input from buttons: ";
        vector::print(input);

        Led led(18);  // Lysdiod på GPIO 18
    #endif
	}

}
