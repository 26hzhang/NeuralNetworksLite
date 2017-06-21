package nd4j.isaac.examples;

import nd4j.isaac.initialization.Weight;
import nd4j.isaac.neuralnetworks.Perceptron;
import nd4j.isaac.utils.GaussianDistribution;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

public class PerceptronExample {

	public static void main(String[] args) {
		// Declare variables and constants for perceptrons
		final int trainNum = 1000; // number of training data
		final int testNum = 200; // number of test data
		final int nIn = 2; // dimensions of input data

		INDArray trainData = Nd4j.create(new double[trainNum * nIn], new int[]{trainNum, nIn});
		INDArray trainLabel = Nd4j.create(new double[trainNum], new int[]{trainNum, 1});

		INDArray testData = Nd4j.create(new double[testNum * nIn], new int[]{testNum, nIn});
		INDArray testLabel = Nd4j.create(new double[testNum], new int[]{testNum, 1});

		final int epochs = 2001; // maximum training epochs
		final double learningRate = 1.0; // learning rate can be 1 in perceptrons

		// Create training data and test data for demo.
		// Let training data set for each class follow Normal (Gaussian) distribution here:
		//   class 1 : x1 ~ N( -2.0, 1.0 ), y1 ~ N( +2.0, 1.0 )
		//   class 2 : x2 ~ N( +2.0, 1.0 ), y2 ~ N( -2.0, 1.0 )
		final Random rng = new Random(1234); // random seed
		GaussianDistribution g1 = new GaussianDistribution(-2.0, 1.0, rng);
		GaussianDistribution g2 = new GaussianDistribution(2.0, 1.0, rng);

		for (int i = 0; i < trainNum / 2 - 1; i++) {
			trainData.put(i, 0, Nd4j.scalar(g1.random()));
			trainData.put(i, 1, Nd4j.scalar(g2.random()));
			trainLabel.put(i, Nd4j.scalar(1));
		}

		for (int i = 0; i < testNum/ 2 - 1; i++) {
			testData.put(i, 0, Nd4j.scalar(g1.random()));
			testData.put(i, 1, Nd4j.scalar(g2.random()));
			testLabel.put(i, Nd4j.scalar(1));
		}

		for (int i = trainNum / 2; i < trainNum; i++) {
			trainData.put(i, 0, Nd4j.scalar(g2.random()));
			trainData.put(i, 1, Nd4j.scalar(g1.random()));
			trainLabel.put(i, Nd4j.scalar(-1));
		}

		for (int i = testNum / 2; i < testNum; i++) {
			testData.put(i, 0, Nd4j.scalar(g2.random()));
			testData.put(i, 1, Nd4j.scalar(g1.random()));
			testLabel.put(i, Nd4j.scalar(-1));
		}

		// Build SingleLayerNeuralNetworks model
		Perceptron classifier = new Perceptron(nIn, Weight.ZERO);

		// train
		for (int epoch = 0; epoch < epochs; epoch++) {
			int classified = 0;
			for (int i = 0; i < trainNum; i++) {
				classified += classifier.train(trainData.getRow(i), trainLabel.getRow(i), learningRate);
			}
			if (classified == trainNum)
				break;
		}

		// predict
		INDArray predictedLabel = classifier.predict(testData);

		// evaluate model
		int[][] confusionMatrix = new int[2][2];
		double accuracy = 0.0;
		double precision = 0.0;
		double recall = 0.0;

		for (int i = 0; i < testNum; i++) {
			if (predictedLabel.getRow(i).getDouble(0) > 0) {
				if (testLabel.getRow(i).getDouble(0) > 0) {
					accuracy += 1;
					precision += 1;
					recall += 1;
					confusionMatrix[0][0] += 1;
				} else {
					confusionMatrix[1][0] += 1;
				}
			} else {
				if (testLabel.getRow(i).getDouble(0) > 0) {
					confusionMatrix[0][1] += 1;
				} else {
					accuracy += 1;
					confusionMatrix[1][1] += 1;
				}
			}
		}
		accuracy /= testNum;
		precision /= confusionMatrix[0][0] + confusionMatrix[1][0];
		recall /= confusionMatrix[0][0] + confusionMatrix[0][1];

		System.out.println("----------------------------");
		System.out.println("Perceptrons model evaluation");
		System.out.println("----------------------------");
		System.out.printf("Accuracy:  %.1f %%\n", accuracy * 100);
		System.out.printf("Precision: %.1f %%\n", precision * 100);
		System.out.printf("Recall:    %.1f %%\n", recall * 100);
		INDArray w = classifier.w;
		System.out.println(w.toString());
	}
}
/*
Perceptrons model evaluation
----------------------------
Accuracy:  99.5 %
Precision: 99.0 %
Recall:    100.0 %
 */