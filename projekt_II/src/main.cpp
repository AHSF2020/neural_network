/**
 * @note The neural network is trained correctly and predicts with 100 % accuracy.
 * 
 *       The code is very clear and easy to follow. You've named variables, 
 *       methods and classes wisely.
 * 
 * @remark The system works as intended, i.e. the LED is enabled for odd numbers of high inputs.
 *         But the neural network isn't used for the prediction. The input signals from the
 *         buttons are supposed to be used as inputs for the neural network, which is supposed
 *         to make a prediction, i.e. you run a feedforward to generate new output values.
 * 
 *         The output of the neural network is then supposed to set the value of the LED.
 *         To solve this, see the segment marked SUGGESTED_SOLUTION below.
 *         Feel free to use it.
 * 
 *         The files `act_func.h`, `neural_network.h`, and `neural_network.cpp` can be removed,
 *         since they're not used in this implementation.
 * 
 *         Some comment blocks are out of date and need to be updated.
 *         Your inline comments are great, but please translate them into English.      
 */

/** @remark Please update this comment. */
/******************************************************************************* 
 * @brief Demonstration of a dense layer network. 
 ******************************************************************************/
#include <iostream>
#include <vector>
#include <thread> // För delay
#include <chrono> // För tidsenhet
#include "dense_layer.h"
#include "button.h"
#include "led.h"
#include "utils.h"

#define ENABLE_DRIVERS

using namespace ml;
using namespace rpi;
using namespace utils;

/** 
 * @remark Please add a comment about the buttons being connected to the network, i.e. the buttons 
 *         being connected to the input and the LED being connected to the output. 
 */
/******************************************************************************* 
 * @brief Creates and initializes a dense layer network, which is trained to 
 *        detect a 4-bit XOR pattern. 
 ******************************************************************************/

int main()
{
    const std::vector<std::vector<double>> input_sets{
        {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 0, 1, 1},
        {0, 1, 0, 0}, {0, 1, 0, 1}, {0, 1, 1, 0}, {0, 1, 1, 1},
        {1, 0, 0, 0}, {1, 0, 0, 1}, {1, 0, 1, 0}, {1, 0, 1, 1},
        {1, 1, 0, 0}, {1, 1, 0, 1}, {1, 1, 1, 0}, {1, 1, 1, 1}
    };

    const std::vector<std::vector<double>> reference_sets{
        {0}, {1}, {1}, {0},
        {1}, {0}, {0}, {1},
        {1}, {0}, {0}, {1},
        {0}, {1}, {1}, {0}
    };

    dense_layer hidden_layer{4U, 4U, act_func::tanh};
    dense_layer output_layer{1U, 4U, act_func::relu};

    dense_layer::train(hidden_layer, output_layer, input_sets, reference_sets, 100000, 0.01);

    dense_layer::print(hidden_layer, output_layer, input_sets, reference_sets);

    #ifdef ENABLE_DRIVERS

    Button button1(23), button2(17), button3(27), button4(22);

    std::vector<Button*> buttons = {
        &button1, &button2, &button3, &button4
    };

    Led led(18); // Lysdiod på GPIO 18 deklareras utanför loopen

    while (true) {
        std::vector<double> input(buttons.size(), 0.0);
        int press_count = 0; // Räknar antalet knapptryck

        for (std::size_t i {}; i < buttons.size(); i++) {
            if (buttons[i]->isPressed()) {
                input[i] = 1;
                press_count++;
            }
        }

        std::cout << "Input from buttons: ";
        vector::print(input);

        /** @note Suggested solution to predict with the values from the buttons: */
        #ifdef SUGGESTED_SOLUTION

            // Perform feedforward to generate new output.
            hidden_layer.feedforward(input);
            output_layer.feedforward(hidden_layer.output());

            // Get new output value rounded to nearest integer.
            const auto output{static_cast<int>(output_layer.output()[0U] + 0.5)};

            // Set new value for the LED.
            led.write(output);

        #else
            // Kontrollera om antalet knapptryck är udda
            if (press_count % 2 != 0) {
                led.write(true); // Tänd lysdioden
            } else {
                led.write(false); // Släck lysdioden
            }
        #endif 

        // Delay på 1 sekund efter varje utskrift
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    #endif
}
