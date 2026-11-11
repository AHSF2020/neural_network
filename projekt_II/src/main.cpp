/*******************************************************************************
 * @brief Trains a dense-layer network to predict the parity of four button
 *        inputs and uses its prediction to control an LED.
 ******************************************************************************/
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include "dense_layer.h"
#include "button.h"
#include "led.h"
#include "utils.h"

#define ENABLE_DRIVERS

using namespace ml;
using namespace rpi;
using namespace utils;

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

    Led led(18);

    while (true) {
        std::vector<double> input(buttons.size(), 0.0);

        for (std::size_t i {}; i < buttons.size(); i++) {
            if (buttons[i]->isPressed()) {
                input[i] = 1;
            }
        }

        std::cout << "Input from buttons: ";
        vector::print(input);

        hidden_layer.feedforward(input);
        output_layer.feedforward(hidden_layer.output());

        const bool output{output_layer.output()[0U] >= 0.5};
        led.write(output);

        // Wait one second before reading the buttons again.
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    #endif
}
