//namespace Ann_GA_Algorithm;

using System;

public class ANN
{
    // fields
    public int[] Layers;
    public int LayersSize;
    public double[][][] weights; // for each edge
    public double[][] biases; // for each neuron
    public int FitnessScore;
    public double treshHold;
    private int neuronsInEachLayer;
    public ANN(int layersSize, double tresh)
    {
        treshHold = tresh;
        Layers = new int[layersSize];
        this.LayersSize = layersSize;
        neuronsInEachLayer = 16;
        InitializeNeurons(neuronsInEachLayer);
        CreateRandomNetwork();
        
    }

    private void CreateRandomNetwork()
    {
        createRandomWeights();
        createRandomBiases();
        
    }

    private void InitializeNeurons(int nuronsInEachLayer)
    {
        for (int i = 0; i < LayersSize; i++)
        {
            Layers[i] = nuronsInEachLayer;
        }
        Layers[LayersSize - 1] = 1;
    }
    /************************************************************************
     * Name:createRandomWeights
     * Description: this method create a random starting weights, i did this
     * for more easy genetic algorithms operation.
     ************************************************************************/
    private void createRandomWeights()
    {
        Random r = new Random();
        // create randomWeights
        weights = new double[LayersSize - 1][][]; // we have layer size -1 edges layer.
        for (int i = 0; i < LayersSize-1; i++)
        {
            weights[i] = new double[Layers[i + 1]][]; // each layer connect to layer +1;
            for (int j = 0; j < Layers[i + 1]; j++)
            {
                weights[i][j] = new double[Layers[i]]; //
                for (int k = 0; k < Layers[i]; k++)
                {
                    weights[i][j][k] = 2 * r.NextDouble() - 1; // this for get weight between -1 to 1
                }
            }
        }
    }
    /************************************************************************
     * Name:createRandomBiases
     * Description: this method create a random starting biases, i did this
     * for more easy genetic algorithms operation.
     ************************************************************************/
    private void createRandomBiases()
    {
        Random r = new Random();
        biases = new double[LayersSize][];
        for (int i = 0; i < LayersSize; i++)
        {
            biases[i] = new double[Layers[i]];
            for (int j = 0; j < Layers[i]; j++)
            {
                biases[i][j] = 2 * r.NextDouble() - 1; 
            }
        }
    }
    /************************************************************************
     * Name:FeedForward
     * Description: simply move the input foward the network and get the output
     * the output can be 1 or 0.
     ************************************************************************/
    public int FeedForward(int[] input)
    {
        double[][] neuronValues = new double[LayersSize][];
        neuronValues[0] = new double[Layers[0]];
        // put the input in the first layer
        for (int i = 0; i < Layers[0]; i++)
        {
            neuronValues[0][i] = input[i];
        }

        // move the input foward
        for (int i = 1; i < LayersSize; i++)
        {
            neuronValues[i] = new double[Layers[i]];
            for (int j = 0; j < Layers[i]; j++)
            {
                double z = 0;
                for (int k = 0; k < Layers[i - 1]; k++)
                {
                    z += neuronValues[i - 1][k] * weights[i - 1][j][k];
                }

                z += biases[i - 1][j];
                neuronValues[i][j] = Sigmoid(z);
            }
        }

        if (neuronValues[LayersSize - 1][0] > treshHold) return 1;
        return 0;

    }
    /************************************************************************
     * Name:Sigmoid
     * Description: simple implementation of sigmoid function.
     ************************************************************************/
    private double Sigmoid(double input)
    {
        return 1 / (1 + Math.Exp(-input));
    }
}