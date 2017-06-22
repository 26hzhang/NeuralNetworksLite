package com.isaac.java.examples;

import com.isaac.java.neuralnetworks.Perceptron;
import com.isaac.java.utils.GaussianDistribution;

import java.util.Random;

public class PerceptronExample {

	public static void main(String[] args) {
		/*
		 * Declare variables and constants for perceptrons
		 */
		final int trainNum = 1000; // number of training data
		final int testNum = 200; // number of test data
		final int nIn = 2; // dimensions of input data
		
		double[][] trainData = new double[trainNum][nIn]; // input data for training
		int[] trainLabel = new int[trainNum]; // output data (label) for training
		
		double[][] testData = new double[testNum][nIn]; // input data for test
		int[] testLabel = new int[testNum]; // lable of inputs
		int[] predictedLabel = new int[testNum]; // output data predicted by the model
		
		final int epochs = 2000; // maximum training epochs
		final double learningRate = 1.0; // learning rate can be set as 1 in perceptrons
		
		/*
		 * Create training data and test data for demo
		 * 
		 * Let training data set for each class follow Normal (Gaussian) distribution here:
		 * 		class 1: x1 ~ N(-2.0, 1.0), y1 ~ N(+2.0, 1.0)
		 * 		class 2: x2 ~ N(+2.0, 1.0), y2 ~ N(-2.0, 1.0)
		 */
		final Random rng = new Random(1234); // seed random, how to choose the random seed?
		GaussianDistribution g1 = new GaussianDistribution(-2.0, 1.0, rng);
		GaussianDistribution g2 = new GaussianDistribution(2.0, 1.0, rng);
		
		// data set in class 1:
		for (int i = 0; i < trainNum / 2 - 1; i++) {
			trainData[i][0] = g1.random();
			trainData[i][1] = g2.random();
			trainLabel[i] = 1;
		}
		for (int i = 0; i < testNum / 2 - 1; i++) {
			testData[i][0] = g1.random();
			testData[i][1] = g2.random();
			testLabel[i] = 1;
		}
		
		// data set in class 2:
		for (int i = trainNum / 2; i < trainNum; i++) {
			trainData[i][0] = g2.random();
			trainData[i][1] = g1.random();
			trainLabel[i] = -1;
		}
		for (int i = testNum / 2; i < testNum; i++) {
			testData[i][0] = g2.random();
			testData[i][1] = g1.random();
			testLabel[i] = -1;
		}
		
		/*
		 * Build Single Layer Neural Networks Model
		 */
		int epoch = 0; // training epoches
		Perceptron classifier = new Perceptron(nIn); // construct perceptrons
		// train models
		while (epoch <= epochs) {
			int classified = 0;
			for (int i = 0; i < trainNum; i++) {
				classified += classifier.train(trainData[i], trainLabel[i], learningRate);
			}
			if (classified == trainNum)
				break; // when all data classified correctly
			epoch++;
		}
		
		/*
		 * test
		 */
		for (int i = 0; i < testNum; i++) {
			predictedLabel[i] = classifier.predict(testData[i]);
		}
		
		/*
		 * Evaluate the model
		 */
		int[][] confusionMatrix = new int[2][2];
		double accuracy = 0.0;
		double precision = 0.0;
		double recall = 0.0;
		for (int i = 0; i < testNum; i++) {
			if (predictedLabel[i] > 0) {
				if (testLabel[i] > 0) {
					accuracy += 1;
					precision += 1;
					recall += 1;
					confusionMatrix[0][0] += 1;
				} else
					confusionMatrix[1][0] += 1;
			} else {
				if (testLabel[i] > 0)
					confusionMatrix[0][1] += 1;
				else {
					accuracy += 1;
					confusionMatrix[1][1] += 1;
				}
			}
		}
		
		accuracy /= testNum;
		precision /= confusionMatrix[0][0] + confusionMatrix[1][0];
		recall /= confusionMatrix[0][0] + confusionMatrix[0][1];
		
		System.out.println(String.format("Perceptrons model evaluation:\n" + 
							"Accuracy: %.1f\nPrecision: %.1f\nRecall: %.1f", accuracy * 100, 
							precision * 100, recall * 100));
		System.out.println("----------------------");
		for (int i = 0; i < classifier.w.length; i++) {
			System.out.print(classifier.w[i] + "\t");
		}
	}

}
/*
Perceptrons model evaluation:
-----------------------------
Accuracy: 99.0
Precision: 98.0
Recall: 100.0
 */