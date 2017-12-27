package com.isaac.examples;

import com.isaac.initialization.WeightInit;
import com.isaac.neuralnetworks.Perceptron;
import com.isaac.utils.GaussianDistribution;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

public class PerceptronExample {

	public static void main(String[] args) {
		// Declare variables and constants for perceptrons
		final int trainSetSize = 1000; // number of training data
		final int testSetSize = 200; // number of test data
		final int nIn = 2; // dimensions of input data

		INDArray trainSet = Nd4j.create(new double[trainSetSize * nIn], new int[]{trainSetSize, nIn});
		INDArray trainLabel = Nd4j.create(new double[trainSetSize], new int[]{trainSetSize, 1});

		INDArray testSet = Nd4j.create(new double[testSetSize * nIn], new int[]{testSetSize, nIn});
		INDArray testLabel = Nd4j.create(new double[testSetSize], new int[]{testSetSize, 1});

		final int epochs = 2001; // maximum training epochs
		final double learningRate = 1.0; // learning rate can be 1 in perceptrons

		// Create training data and test data for demo.
		// Let training data set for each class follow Normal (Gaussian) distribution here:
		//   class 1 : x1 ~ N( -2.0, 1.0 ), y1 ~ N( +2.0, 1.0 )
		//   class 2 : x2 ~ N( +2.0, 1.0 ), y2 ~ N( -2.0, 1.0 )
		final Random rng = new Random(1234); // random seed
		GaussianDistribution g1 = new GaussianDistribution(-2.0, 1.0, rng);
		GaussianDistribution g2 = new GaussianDistribution(2.0, 1.0, rng);

		for (int i = 0; i < trainSetSize / 2 - 1; i++) {
			trainSet.put(i, 0, Nd4j.scalar(g1.random()));
			trainSet.put(i, 1, Nd4j.scalar(g2.random()));
			trainLabel.put(i, Nd4j.scalar(1));
		}

		for (int i = 0; i < testSetSize/ 2 - 1; i++) {
			testSet.put(i, 0, Nd4j.scalar(g1.random()));
			testSet.put(i, 1, Nd4j.scalar(g2.random()));
			testLabel.put(i, Nd4j.scalar(1));
		}

		for (int i = trainSetSize / 2; i < trainSetSize; i++) {
			trainSet.put(i, 0, Nd4j.scalar(g2.random()));
			trainSet.put(i, 1, Nd4j.scalar(g1.random()));
			trainLabel.put(i, Nd4j.scalar(-1));
		}

		for (int i = testSetSize / 2; i < testSetSize; i++) {
			testSet.put(i, 0, Nd4j.scalar(g2.random()));
			testSet.put(i, 1, Nd4j.scalar(g1.random()));
			testLabel.put(i, Nd4j.scalar(-1));
		}

		// Build SingleLayerNeuralNetworks model
		Perceptron classifier = new Perceptron(nIn, WeightInit.ZERO);

		// train
		for (int epoch = 0; epoch < epochs; epoch++) {
			int classified = 0;
			for (int i = 0; i < trainSetSize; i++) {
				classified += classifier.train(trainSet.getRow(i), trainLabel.getRow(i), learningRate);
			}
			if (classified == trainSetSize) break;
		}

		// predict
		INDArray predictedLabel = classifier.predict(testSet);

		// evaluate model
		int[][] confusionMatrix = new int[2][2];
		double accuracy = 0.0;
		double precision = 0.0;
		double recall = 0.0;

		for (int i = 0; i < testSetSize; i++) {
			if (predictedLabel.getRow(i).getDouble(0) > 0) {
				if (testLabel.getRow(i).getDouble(0) > 0) {
					accuracy += 1;
					precision += 1;
					recall += 1;
					confusionMatrix[0][0] += 1;
				} else { confusionMatrix[1][0] += 1; }
			} else {
				if (testLabel.getRow(i).getDouble(0) > 0) { confusionMatrix[0][1] += 1; }
				else {
					accuracy += 1;
					confusionMatrix[1][1] += 1;
				}
			}
		}
		accuracy /= testSetSize;
		precision /= confusionMatrix[0][0] + confusionMatrix[1][0];
		recall /= confusionMatrix[0][0] + confusionMatrix[0][1];

		System.out.println("Perceptrons model evaluation");
		System.out.println("----------------------------");
		System.out.printf("Accuracy:  %.1f %%\n", accuracy * 100);
		System.out.printf("Precision: %.1f %%\n", precision * 100);
		System.out.printf("Recall:    %.1f %%\n", recall * 100);
		INDArray w = classifier.getW();
		System.out.println(w.toString());
	}
}