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

    while (true) {
        std::vector<double> input(buttons.size(), 0.0);
        for (std::size_t i {}; i < buttons.size(); i++) {
            if (buttons[i]->isPressed()) {
                input[i] = 1;
            }
        }

        std::cout << "Input from buttons: ";
        vector::print(input);

        // Delay på 1 sekund efter varje utskrift
        std::this_thread::sleep_for(std::chrono::seconds(1));

        Led led(18); // Lysdiod på GPIO 18
    }
    #endif
}
