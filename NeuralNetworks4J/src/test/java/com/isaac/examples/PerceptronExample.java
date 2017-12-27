package com.isaac.examples;

import com.isaac.neuralnetworks.Perceptron;
import com.isaac.utils.GaussianDistribution;

import java.util.Random;

public class PerceptronExample {

	public static void main(String[] args) {
		// Declare variables and constants for perceptrons
		final int trainSetSize = 1000; // number of training data
		final int testSetSize = 200; // number of test data
		final int nIn = 2; // dimensions of input data
		
		double[][] trainSet = new double[trainSetSize][nIn]; // input data for training
		int[] trainLabel = new int[trainSetSize]; // output data (label) for training
		
		double[][] testSet = new double[testSetSize][nIn]; // input data for test
		int[] testLabel = new int[testSetSize]; // label of inputs
		int[] predictLabel = new int[testSetSize]; // output data predicted by the model
		
		final int epochs = 2000; // maximum training epochs
		final double learningRate = 1.0; // learning rate can be set as 1 in perceptrons
		
		/*
		 * Create training data and test data for demo
		 * Let training data set for each class follow Normal (Gaussian) distribution here:
		 * 		class 1: x1 ~ N(-2.0, 1.0), y1 ~ N(+2.0, 1.0)
		 * 		class 2: x2 ~ N(+2.0, 1.0), y2 ~ N(-2.0, 1.0)
		 */
		final Random rng = new Random(1234); // seed random, how to choose the random seed?
		GaussianDistribution g1 = new GaussianDistribution(-2.0, 1.0, rng);
		GaussianDistribution g2 = new GaussianDistribution(2.0, 1.0, rng);
		
		// data set in class 1:
		for (int i = 0; i < trainSetSize / 2 - 1; i++) {
			trainSet[i][0] = g1.random();
			trainSet[i][1] = g2.random();
			trainLabel[i] = 1;
		}
		for (int i = 0; i < testSetSize / 2 - 1; i++) {
			testSet[i][0] = g1.random();
			testSet[i][1] = g2.random();
			testLabel[i] = 1;
		}
		
		// data set in class 2:
		for (int i = trainSetSize / 2; i < trainSetSize; i++) {
			trainSet[i][0] = g2.random();
			trainSet[i][1] = g1.random();
			trainLabel[i] = -1;
		}
		for (int i = testSetSize / 2; i < testSetSize; i++) {
			testSet[i][0] = g2.random();
			testSet[i][1] = g1.random();
			testLabel[i] = -1;
		}
		
		// Build Single Layer Neural Networks Model
		Perceptron classifier = new Perceptron(nIn); // construct perceptrons

		// train models
		int epoch = 0; // training epochs
		while (epoch <= epochs) {
			int classified = 0;
			for (int i = 0; i < trainSetSize; i++) classified += classifier.train(trainSet[i], trainLabel[i], learningRate);
			if (classified == trainSetSize) break; // when all data classified correctly
			epoch++;
		}
		
		// test
		for (int i = 0; i < testSetSize; i++) {
			predictLabel[i] = classifier.predict(testSet[i]);
		}
		
		// Evaluate the model
		int[][] confusionMatrix = new int[2][2];
		double accuracy = 0.0;
		double precision = 0.0;
		double recall = 0.0;
		for (int i = 0; i < testSetSize; i++) {
			if (predictLabel[i] > 0) {
				if (testLabel[i] > 0) {
					accuracy += 1;
					precision += 1;
					recall += 1;
					confusionMatrix[0][0] += 1;
				} else confusionMatrix[1][0] += 1;
			} else {
				if (testLabel[i] > 0) confusionMatrix[0][1] += 1;
				else {
					accuracy += 1;
					confusionMatrix[1][1] += 1;
				}
			}
		}
		
		accuracy /= testSetSize;
		precision /= confusionMatrix[0][0] + confusionMatrix[1][0];
		recall /= confusionMatrix[0][0] + confusionMatrix[0][1];
		
		System.out.println(String.format("Perceptrons model evaluation:\n" + 
							"Accuracy: %.1f\nPrecision: %.1f\nRecall: %.1f", accuracy * 100, 
							precision * 100, recall * 100));
		System.out.println("----------------------");
		for (int i = 0; i < classifier.getW().length; i++) {
			System.out.print(classifier.getW()[i] + "\t");
		}
	}

}