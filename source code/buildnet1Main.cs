﻿// See https://aka.ms/new-console-template for more information

Console.WriteLine("Enter the train file name: ");
string filename = Console.ReadLine();
Console.WriteLine("Enter the test file name, or enter 'None' so the program will divide the train file to test and train: ");
string testFile = Console.ReadLine();
Console.WriteLine("Please wait... this might take 15-30 minutes");


// string filename = "nn0.txt";
string filepath = Path.Combine(AppContext.BaseDirectory,"InputFilesForBuildNet", filename);
string fileOutput = "wnet1.txt";
string fileOutputPath = Path.Combine(AppContext.BaseDirectory,"InputFilesForRunNet", fileOutput);

InputData inputData = null;
if(testFile == "None")  inputData = IO.readFile(filepath,0.25,filepath,false);
else  inputData = IO.readFile(filepath,0,filepath,true);

// Initialize the genetic algorithm with your parameters
GeneticAlgorithm ga = new GeneticAlgorithm(
    populationSize: 500,
    mutationRate: 0.02,
    crossOverRate: 0.75,
    maxGen: 15000,
    id: inputData,
    treshHold: 0.5,
    layerSize: 6
);

// Run the algorithm
ga.Evolve(maxGeneration: 200, selectionSize: 15);

// After the algorithm has run, write the best network to a file
IO.writeNetworkToFile(ga.bestSulotion, fileOutputPath);

