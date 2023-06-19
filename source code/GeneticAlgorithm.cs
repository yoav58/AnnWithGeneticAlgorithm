using System;
using System.Collections.Generic;
using System.Linq;

//namespace Ann_GA_Algorithm;

public class GeneticAlgorithm
{
    private int populationSize;
    private List<ANN> candidateSolutions;
    public ANN bestSulotion;
    public int fitnessFunctionCalls;
    private int generationWithoutImprovment;
    private int currentSize;
    private int maxSize;
    public bool shouldRun;
    private double mutationRate;
    private InputData inputData;
    private double crossOverRate;
    private int maxGeneration; 
    private double treshHoldAnn;
    private int numberOfLayers;


    public GeneticAlgorithm(int populationSize, double mutationRate, double crossOverRate, int maxGen,InputData id,double treshHold, int layerSize)
    {
        candidateSolutions = new List<ANN>();
        this.populationSize = populationSize;
        this.mutationRate = mutationRate;
        this.crossOverRate = crossOverRate;
        this.maxSize = maxGen;
        this.inputData = id;
        this.numberOfLayers = layerSize;
        this.treshHoldAnn = treshHold;
    }

    /************************************************************************
     * Name:InitializePopulation
    * Description: create random population.
    ************************************************************************/
    private void InitializePopulation()
    {
        for (int i = 0; i < populationSize; i++)
        {
            candidateSolutions.Add(new ANN(numberOfLayers,treshHoldAnn));
        }
    }
    /************************************************************************
     * Name:CalculateFitness
    * Description: this method take Single network and calculate her fitness
     * score, the score is simply the number of accurate labels
    ************************************************************************/
    private void CalculateFitness(ANN ann)
    {
        Interlocked.Increment(ref fitnessFunctionCalls);//fitnessFunctionCalls++;
        int score = 0;
        foreach (var example in inputData.TrainData)
        {
            int predict = ann.FeedForward(example.Item1);
            if (predict == example.Item2) score++;
        }

        ann.FitnessScore = score;
    }
    /************************************************************************
     * Name:FitnessFunction
    * Description: the general fitness function.
    ************************************************************************/
    public void FitnessFunction()
    {
        Parallel.ForEach(candidateSolutions, network =>
        {
            CalculateFitness(network);
        });
        
        //foreach (var network in candidateSolutions)
        //{
          //  CalculateFitness(network);
       // }
    }
    /************************************************************************
     * Name:CandidateSelection
    * Description: this function is to find parent to the next generation,
     * we first choose randomly k candidate and then choose the best from them.
    ************************************************************************/
    public ANN CandidateSelection(int size)
    {
        List<ANN> candidateToNextGeneration = new List<ANN>();
        Random r = new Random();
        int genSize = candidateSolutions.Count;
        for(int i = 0; i < size; i++) candidateToNextGeneration.Add(candidateSolutions[r.Next(size)]);
        ANN parent = candidateToNextGeneration[0];
        foreach (var c in candidateToNextGeneration) parent = c.FitnessScore > parent.FitnessScore ? c : parent;
        return parent;
    }
    /************************************************************************
     * Name:CrossOver
    * Description: this method do cross over between two parents and create
     * their son.
    ************************************************************************/
    public ANN CrossOver(ANN parent1, ANN parent2)
    {
        Random r = new Random();
        if (r.NextDouble() > crossOverRate) return r.NextDouble() < 0.5 ? parent1 : parent2;
        ANN child = new ANN(parent1.LayersSize, treshHoldAnn);
        
        // start crossover weights
        crossOverWeights(parent1,parent2,child);
        crossOverBiases(parent1, parent2, child);
        return child;
    }
    /************************************************************************
    * Name:crossOverWeights
    * Description: this method update the weights of the son. by take
     * random weight from father1 and random weights from father 2.
    ************************************************************************/
    private void crossOverWeights(ANN parent1, ANN parent2,ANN child){
        for (int i = 0; i < child.weights.Length; i++)
        {
            for (int j = 0; j < child.weights[i].Length; j++)
            {
                for (int k = 0; k < child.weights[i][j].Length; k++)
                {
                    double value = (new Random().NextDouble() < 0.5)
                        ? parent1.weights[i][j][k]
                        : parent2.weights[i][j][k];
                    child.weights[i][j][k] = value;
                }
            }
        }
        
    }
    /************************************************************************
    * Name:crossOverBiases
    * Description: this method update the biases of the son. by take
    * random biases from father1 and random biases from father 2.
    ************************************************************************/ 
    private void crossOverBiases(ANN parent1, ANN parent2, ANN child)
    {
        for (int i = 0; i < child.biases.Length; i++)
        {
            for (int j = 0; j < child.biases[i].Length; j++)
            {
                double value = (new Random().NextDouble() < 0.5) ? parent1.biases[i][j] : parent2.biases[i][j];
                child.biases[i][j] = value;
            }
        }
    }
    /************************************************************************
    * Name:Mutate
    * Description: this method simply create a mutate at the child.
    ************************************************************************/ 
    public void Mutate(ANN child)
    {
        // mutate some of the weights
        MutateWeights(child);
        MutateBiases(child);
        
    }
    /************************************************************************
    * Name:MutateWeights
    * Description: by the mutate rate, i choose if change each weight.
    ************************************************************************/ 
    private void MutateWeights(ANN child)
    {
        Random r = new Random();
        for (int i = 0; i < child.weights.Length; i++)
        {
            for (int j = 0; j < child.weights[i].Length; j++)
            {
                for (int k = 0; k < child.weights[i][j].Length; k++)
                {
                    if (r.NextDouble() < mutationRate)
                    {
                        child.weights[i][j][k] += r.NextDouble() * 2 - 1; // Adding a random value between -1 and 1
                    }
                }
            }
        }
    }
    /************************************************************************
    * Name:MutateBiases
    * Description: by the mutate rate, i choose if change each bias.
    ************************************************************************/     
    private void MutateBiases(ANN child)
    {
        Random r = new Random();
        for (int i = 0; i < child.biases.Length; i++)
        {
            for (int j = 0; j < child.biases[i].Length; j++)
            {
                if (r.NextDouble() < mutationRate)
                {
                    child.biases[i][j] += r.NextDouble() * 2 - 1; // Adding a random value between -1 and 1
                }
            }
        }
    }
    /************************************************************************
    * Name:Evolve
    * Description: run the genetic algorithm, start create generations and
     * do all the operations of genetic algorithm.
    ************************************************************************/  
    public void Evolve(int maxGeneration, int selectionSize)
    {
        // first step
        InitializePopulation();
        FitnessFunction();
        
        // start evolve generations
        for (int gen = 0; gen < maxGeneration; gen++)
        {
            List<ANN> nextGeneration = new List<ANN>();
            EvolveNextGeneration(nextGeneration,maxGeneration,selectionSize);
            candidateSolutions = nextGeneration;
            FitnessFunction();
            ANN currentBest = candidateSolutions.OrderBy(network => network.FitnessScore).Last();
            if (bestSulotion == null || currentBest.FitnessScore > bestSulotion.FitnessScore)
                bestSulotion = currentBest;
        }

        Console.WriteLine("best score accuracy is " + FinalTest(bestSulotion) +"%");
        Console.WriteLine("the number of calls to fitness function: "+ fitnessFunctionCalls);

    }
    /************************************************************************
    * Name:EvolveNextGeneration
    * Description: create a new genenration.
    ************************************************************************/  
    private void EvolveNextGeneration(List<ANN> nextGeneration,int maxGeneration, int selectionSize)
    {
        for (int i = 0; i < populationSize; i++)
        {
            ANN parent1 = CandidateSelection(selectionSize);
            ANN parent2 = CandidateSelection(selectionSize);
            ANN child = CrossOver(parent1, parent2);
            Mutate(child);
            nextGeneration.Add(child);
        }
    }

    private double FinalTest(ANN ann)
    {
        int score = 0;
        foreach (var example in inputData.TestData)
        {
            int predict = ann.FeedForward(example.Item1);
            if (predict == example.Item2) score++;
        }

        double accuracy = ((double)score / (double)inputData.TestData.Count) * 100;
        return accuracy;
    }
}