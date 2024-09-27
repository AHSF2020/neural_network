#pragma once
#include <iostream>
#include <vector>

namespace ml

{

    class dense_layer
    {

        public:

            // konstruktor
            dense_layer() = delete;     // Tar bort default inställningar i konstruktor
            dense_layer(const std::size_t node_count , const std::size_t weight_count);


            // Getter-metoder -- ge värden
            std::size_t node_count() const;         // Getter-metoder från konstruktorn
            std::size_t weight_count() const;       // Getter-metoder från konstruktorn 

            
            const std::vector<double>& bias();       // Getter-metoder från Private:
            const std::vector<double>& weight();     // Getter-metoder från Private:
            const std::vector<double>& error();      // Getter-metoder från Private:
            const std::vector<double>& output();     // Getter-metoder från Private:

            const std::vector<std::vector<double>& myWeights() const;  // vector holding the weights of each node


            // Set-metoder -- men väljer ej att använda dem då beröäkningarna görs private


            // Metoder för prototyper "yani funktionsprotyper". 



        private:
            std::vector<double> myOutput;    // Vector holding the output of each node.
            std::vector<double> myError;     // Vector holding the calculated error of each node.
            std::vector<double> myBias;      // Vector holding the bias of each node.
            std::vector<double> myWeight;    // Vector holding the weights of each node.

            std::vector<std::vector<double> myWeights;  // vector holding the weights of each node

         // Set-metoder


    } // end class dens_layer

} // end namespace