using System;
using System.Collections.Generic;

//namespace Ann_GA_Algorithm;

// this is class that take the input
public class InputData
{
    public List<Tuple<int[], int>> TrainData { get; set; }
    public List<Tuple<int[], int>> TestData { get; set; }
    
    public InputData(List<Tuple<int[], int>> tr, List<Tuple<int[], int>> te){
        this.TrainData = tr;
        this.TestData = te;
    }
}