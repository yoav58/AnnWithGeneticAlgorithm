// See https://aka.ms/new-console-template for more information

Console.WriteLine("start program");

string fileName = "wnet1.txt";
string testName = "testnet1.txt";
string testPath = Path.Combine(AppContext.BaseDirectory,"InputForRunnet1", testName);
string filepath = Path.Combine(AppContext.BaseDirectory,"InputForRunnet1", fileName);
InputData inputData = IO.readFileTest(testPath);
ANN ann = IO.ReadNetworkFromFile(filepath);
int[] outputPredictions = new int[inputData.TestData.Count];

double accuracy = 0;
for(int i = 0; i < inputData.TestData.Count; i++) {
    outputPredictions[i] = ann.FeedForward(inputData.TestData[i].Item1);
    //if (outputPredictions[i] == inputData.TestData[i].Item2) ++accuracy;
}

//double result = (accuracy / inputData.TestData.Count) * 100;
//Console.WriteLine("the accuracy is " +result );
string outputFile = "AnnOutput.txt";
string outPutPath = Path.Combine(AppContext.BaseDirectory,"OutputForRunnet1", outputFile);
IO.WriteOutputData(outPutPath, outputPredictions);

Console.WriteLine("program end");

Console.WriteLine("Press any key to exit...");
Console.ReadKey();