#pragma once
#include <iostream>
#include <vector>

namespace ml
{

    enum class act_func
    {
        relu,
        tanh
    };

    class dense_layer
    {
        public:
            /**
             * @brief Creates new dense layer.
             * 
             * @param node_count The number of nodes in the layer.
             * @param weight_count The number of weights per node in the layer.
             * @param act_func The activation function to use for the layer (default = ReLU):
             * @param myOutput Vector holding the output of each node.
             * @param myError Vector holding the calculated error of each node.
             * @param myBias // Vector holding the bias of each node.
             * @param myWeights // vector holding the weights of each node
             */

            // Konstruktor
            dense_layer() = delete;     // Tar bort default inställningar i konstruktor
            dense_layer(const std::size_t node_count , const std::size_t weight_count,
                        const act_func act_func = act_func::relu);

            
            //-----------Getter-metoder -- ge värden------------------
            /**
             * @brief Provides the number of nodes in the layer.
             * 
             * @return The number of nodes as an integer.
             */
            std::size_t node_count() const;         // Getter-metoder från konstruktorn
            std::size_t weight_count() const;       // Getter-metoder från konstruktorn 

            const std::vector<double>& bias() const;       // Getter-metoder från Private:
            const std::vector<std::vector<double>>& weights() const; // Getter-metoder från Private:
            const std::vector<double>& error() const;      // Getter-metoder från Private:
            const std::vector<double>& output() const;     // Getter-metoder från Private:

           
            void feedforward(const std::vector<double>& input); // Beräknar nya utsignaler i hela lagret. 

            void backpropagate(const std::vector<double>& reference); // Backpropagation för output-lager (vi beräknar fel i lagret).
            
            void backpropagate(const dense_layer& next_layer); // Backpropagation för det dolda lagret (vi beräknar fel i lagret).
            
            void optimize(const std::vector<double>& input, const double learning_rate); // Justerar vikter och bias.
            
            static void train(dense_layer& hidden_layer, dense_layer& output_layer, 
                              const std::vector<std::vector<double>>& input_sets,
                              const std::vector<std::vector<double>>& reference_sets,
                              std::size_t epoch_count, const double learning_rate = 0.01);
            
            static void print(dense_layer& hidden_layer,dense_layer& output_layer, const std::vector<std::vector<double>>& input_sets,
                                         const std::vector<std::vector<double>>& reference_sets); // print funktion
            //-----------END Getter-metoder ------------------

        private:

            std::vector<double> myOutput;                // Vector holding the output of each node.
            std::vector<double> myError;                 // Vector holding the calculated error of each node.
            std::vector<double> myBias;                  // Vector holding the bias of each node.
            std::vector<std::vector<double>> myWeights;  // vector holding the weights of each node
            act_func myActFunc;                          // Activation function used for the layer.

            double act_func_output(const double number) const;
            double act_func_gradient(const double number) const;

    }; // end class dens_layer

} // end namespace