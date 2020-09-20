#import "SimpleMultiLayerPerceptrons.h"
#import <iostream>

double Neuron::rate;

int main ()
{
	Neuron::rate = 0.1;
	Network Net({2,5,1});

	std::vector< std::vector<double> > Data = {  {0,0,0},
                                        {0,1,1},
                                        {1,0,1},
                                        {1,1,0} };

	std::cout << "Show Prediction before training:" << std::endl;
	for (auto & A : Data) {
		Net.Feed({A[0],A[1]});
		std::cout << "Input: " << "(" << A[0] << "," << A[1] << ") ";
		std::cout << "Output: " << Net.Layers.back()[0].A << std::endl;
	}

	std::cout << "Training..." << std::endl;
	for (int epoch(0);epoch<100000;++epoch) {
		for (auto & A : Data) {
			Net.Feed({A[0],A[1]});
			Net.Back( { A[2] } );
		}
	}

	std::cout << "Show Prediction after training:" << std::endl;
	for (auto& A : Data) {
		Net.Feed({A[0],A[1]});
		std::cout << "Input: " << "(" << A[0] << "," << A[1] << ") ";
		std::cout << "Output: " << Net.Layers.back()[0].A << std::endl;
	}

	return 0;
}
