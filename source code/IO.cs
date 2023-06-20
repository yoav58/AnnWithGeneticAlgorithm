//namespace Ann_GA_Algorithm;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Globalization;

//using Ann_GA_Algorithm;

public class IO
{
    /************************************************************************
     * Name:InputData
     * Description: this method read the learning file, divide to train and
     * test.
     ************************************************************************/
    public static InputData readFile(string filePath, double testRatio, string testPath,bool includeTest)
    {
        List<Tuple<int[], int>> inputData = new List<Tuple<int[], int>>();
    
        // save the data in the right format
        foreach (var line in File.ReadLines(filePath))
        {
            var splitted =  line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            var exampleString = splitted[0];
            var label = int.Parse(splitted[1]);
            var features = exampleString.Select(ch => int.Parse(ch.ToString())).ToArray();
            inputData.Add(new Tuple<int[], int>(features, label));
        }

        // shuffle the data to ensure that we divide to test and train randomly
        int size = inputData.Count -1;
        Random r = new Random();
        while (size > 1)
        {
            int randomPlace = r.Next(size);
            (inputData[size], inputData[randomPlace]) =
                (inputData[randomPlace], inputData[size]); // special c# syntax to swap
            size--;
        }


        // split to test and train
        size = inputData.Count;
        int testPercentage = (int)(size * testRatio);
        var testData = inputData.Take(testPercentage).ToList();
        var trainData = inputData.Skip(testPercentage).ToList();
        if (includeTest)
        {
            trainData = ReadTestFile(testPath);
        }
        InputData input = new InputData(trainData, testData);
        return input;

    }

    private static  List<Tuple<int[], int>> ReadTestFile(string testPath)
    {
        List<Tuple<int[], int>> inputData = new List<Tuple<int[], int>>();
    
        // save the data in the right format
        foreach (var line in File.ReadLines(testPath))
        {
            var splitted =  line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            var exampleString = splitted[0];
            var label = int.Parse(splitted[1]);
            var features = exampleString.Select(ch => int.Parse(ch.ToString())).ToArray();
            inputData.Add(new Tuple<int[], int>(features, label));
        }

        return inputData;
    }
    /************************************************************************
     * Name:writeNetworkToFile
     * Description: this method write the network to txt file.
     ************************************************************************/
    public static void writeNetworkToFile(ANN network, string fileName)
    {
        using (StreamWriter sw = new StreamWriter(fileName))
        {
            // first write the number of layers and tresh hold
            sw.WriteLine(network.LayersSize);
            sw.WriteLine(network.treshHold);


            // Write weights
            for (int i = 0; i < network.weights.Length; i++)
            {
                for (int j = 0; j < network.weights[i].Length; j++)
                {
                    for (int k = 0; k < network.weights[i][j].Length; k++)
                    {
                        sw.Write(network.weights[i][j][k] + " ");
                    }

                    sw.WriteLine();
                }

                sw.WriteLine();
            }

            // Write biases
            for (int i = 0; i < network.biases.Length; i++)
            {
                for (int j = 0; j < network.biases[i].Length; j++)
                {
                    sw.Write(network.biases[i][j] + " ");
                }

                sw.WriteLine();
            }
        }

    }

    /************************************************************************
     * Name:ReadNetworkFromFile
     * Description: this method read the network from txt file.
     ************************************************************************/
    public static ANN ReadNetworkFromFile(string fileName)
    {
        
        using (StreamReader sr = new StreamReader(fileName))
        {
            // Read number of layers
            int layerSize = int.Parse(sr.ReadLine());

            // Read the threshold
            double threshold = double.Parse(sr.ReadLine());

            // Create new ANN
            ANN network = new ANN(layerSize, threshold);

            // Read weights
            for (int i = 0; i < network.weights.Length; i++)
            {
                for (int j = 0; j < network.weights[i].Length; j++)
                {
                    string[] weights = sr.ReadLine().Split(' ');
                    for (int k = 0; k < network.weights[i][j].Length; k++)
                    {

                        string cleaned = new string(weights[k].Where(c => char.IsDigit(c) || c == '.' || c == '-' || c == 'E' || c == 'e').ToArray());
                        network.weights[i][j][k] = double.Parse(cleaned);
                        //network.weights[i][j][k] = double.Parse(weights[k]);
                    }
                }

                sr.ReadLine(); // Skip empty line
            }

            // Read biases
            for (int i = 0; i < network.biases.Length; i++)
            {
                string[] biases = sr.ReadLine().Split(' ');
                for (int j = 0; j < network.biases[i].Length; j++)
                {
                    string cleaned = new string(biases[j].Where(c => char.IsDigit(c) || c == '.' || c == '-' || c == 'E' || c == 'e').ToArray());
                    network.biases[i][j] = double.Parse(cleaned);
                    //network.biases[i][j] = double.Parse(biases[j]);
                }
            }

            return network;
        }

    }

    public static void WriteOutputData(string fileName, int[] predictions)
    {
        using (StreamWriter sw = new StreamWriter(fileName))
        {
            foreach (var prediction in predictions)
            {
                sw.WriteLine(prediction);
            }
        }
    }
    
    public static InputData readFileTest(string filePath)
    {
        List<Tuple<int[], int>> inputData = new List<Tuple<int[], int>>();
    
        // save the data in the right format
        foreach (var line in File.ReadLines(filePath))
        {
            var splitted =  line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            var exampleString = splitted[0];
            int label = 0;
            if(splitted.Length > 1) label = int.Parse(splitted[1]);
            var features = exampleString.Select(ch => int.Parse(ch.ToString())).ToArray();
            inputData.Add(new Tuple<int[], int>(features, label));
        }
        var testData = inputData;
        InputData input = new InputData(null, testData);
        return input;

    }
}