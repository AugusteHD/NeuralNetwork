#include <math.h>					// for exponential function
#include <vector>
#include <array>
#include <map>
#include <assert.h>

/*! \class Neuron (aka Perceptron)
 *  \brief Implements sigmoid activation and gradient backtracking.
 */
class Neuron
{
    friend class Network;
    public:			//!< Object attributes
		double A;       //!< A = sigmoid(Z), where Z is linear sum of inputs
		double Der;     //!< partial derivative with resp. to Z
		double bias;    //!< bias
		std::vector< Neuron* > son;			//!< List of sons
		std::map< Neuron* , double > father; //!< list of fathers with weights

    public:			// Class attributes and methods
		static double rate;  // Learning rate

    private:
        /*! \brief sigmoid. */
		static double sigmoid(double z)
		{
		    return 1/(1+ exp(-z));
		}

		/*! \brief sigmoid derivative expressed with A, i.e sigmoid'(z) = sigmoid(z) * (1-sigmoid'(z)). */
		static double sigmoidprime(double a) {
			return a*(1-a);
		}

		static void Connexion(Neuron & F,Neuron & S,double Weight) //!< Connect 2 neurons
		{
            F.son.push_back(&S);
            S.father[&F] = Weight;
		}

    public:			// Object methods
		Neuron() //// Constructor method
            : A(0),
			  bias(0),
			  Der(0),
			  father(),
			  son()
    {}

    void SetA() {             		//// Set A for Neuron in hidden layers
        double Z = bias;
        for (const auto & F : father)
        {
            Z += F.second * (F.first)->A;
        }
        A = sigmoid(Z);
    }

    void Back(double Target) { 	//// Set Der and update bias + weights for Neurons in the last layer
      Der = (A - Target) * sigmoidprime(A); 		// Set Der

      bias -= Der * rate;												// Update bias
      for (auto& F : father) {
        F.second -= Der * (F.first)->A * rate;	// Update weight
      }
    }

    void Back() {		//// Set Der and update bias + weights for Neurons in hidden layers
      Der=0;
      for (Neuron* S : son) {
        Der += S->Der * sigmoidprime(A) * S->father[this] ; // Calculate Der
      }

      bias -= Der * rate;												// Update bias
      for (auto& F : father) {
        F.second -= Der * (F.first)->A * rate;	// Update weight
      }
    }
};

/*! \class Network (aka multi layer perceptrons)
 *  \brief Implements feed forward and gradient backtracking.
 */
class Network
{
	public:
		static double RNG(){	// Random Number Generator between -1 and 1
			return (rand() % 10000 - 5000.0)/5000.0;
		}

	public:			// Object attribute
		std::vector< std::vector<Neuron> > Layers;	// vector of layers (a layer is a vector of Neurons )

    public:			// Object methods
		Network( std::vector<int> LayerStructure ) {	//// Constructor method (fully connected layers)
			Layers.push_back( std::vector<Neuron>( LayerStructure[0],Neuron() ) );		// Append the first layer

			for (int l(1);l<LayerStructure.size();++l) {
				Layers.push_back( std::vector<Neuron>( LayerStructure[l],Neuron() ) );	// Append a layer
				for (Neuron &N : Layers[l-1])														// for all Neurons in the previous layer
					for (Neuron &M : Layers[l])														// for all Neurons in the curent layer
						Neuron::Connexion( N,M, RNG()/Layers[l-1].size() ); // Connect N and M with random weight
			}
    }

    /*! \brief Feedfoward. */
    void Feed( std::vector<double> INPUT)
    {
        assert (INPUT.size() == Layers[0].size() ); // assert size of INPUT is equal to size of the first layer
        for (int i=0;i<INPUT.size();++i)
            Layers[0][i].A = INPUT[i];							// Set Inputs

        for (int l(1);l<Layers.size();++l)				// for each layer accept the first one
            for (Neuron& N : Layers[l])						// for each Neuron in that layer															// Update Z
                N.SetA();															// Update A
    }

    /*! \brief Backtrack. */
    void Back( std::vector<double> TARGET )
    {   //// Update all biases + weights
        assert (TARGET.size() == Layers.back().size() );	// assert the size of TARGET
																												// is equal to size of the last layer
        for (int k=0; k<TARGET.size();++k )
            Layers.back()[k].Back( TARGET[k] );							// Update values for the last layer

        for (int l=Layers.size()-2;l>0;--l)		// for each layer backward
            for (Neuron & N : Layers[l])						// for each Neuron in that layer
                N.Back();									// Update
    }
};
