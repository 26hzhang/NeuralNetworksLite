package com.isaac.examples;

import com.isaac.layers.LogisticLayer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class LogisticRegressionXORExample {

	public static void main(String[] args) {
		final Random rng = new Random(1234); // seed random
		// Declare variables and constants
		final int patterns = 2; // number of classes, nOut
		final int trainSetSize = 4;
		final int testSetSize = 4;
		final int nIn = 2;

		double[][] trainSet;
		int[][] trainLabel;

		double[][] testSet;
		Integer[][] testLabel;
		Integer[][] predictLabel = new Integer[testSetSize][patterns];

		final int epochs = 2000;
		double learningRate = 0.2;

		int minibatchSize = 1; //  set 1 for on-line training
		int minibatchNumber = trainSetSize / minibatchSize;

		double[][][] train_X_minibatch = new double[minibatchNumber][minibatchSize][nIn];
		int[][][] train_T_minibatch = new int[minibatchNumber][minibatchSize][patterns];
		List<Integer> minibatchIndex = new ArrayList<>();
		for (int i = 0; i < trainSetSize; i++) minibatchIndex.add(i);
		Collections.shuffle(minibatchIndex, rng);

		/*
		 * Training data for demo
		 * class 1 : [0, 0], [1, 1] for negative class
		 * class 2 : [0, 1], [1, 0] for positive class
		 */
		trainSet = new double[][] { { 0., 0. }, { 0., 1. }, { 1., 0. }, { 1., 1. } };
		trainLabel = new int[][] { { 0, 1 }, { 1, 0 }, { 1, 0 }, { 0, 1 } };
		testSet = new double[][] { { 0., 0. }, { 0., 1. }, { 1., 0. }, { 1., 1. } };
		testLabel = new Integer[][] { { 0, 1 }, { 1, 0 }, { 1, 0 }, { 0, 1 } };

		// create minibatches
		for (int i = 0; i < minibatchNumber; i++) {
			for (int j = 0; j < minibatchSize; j++) {
				train_X_minibatch[i][j] = trainSet[minibatchIndex.get(i * minibatchSize + j)];
				train_T_minibatch[i][j] = trainLabel[minibatchIndex.get(i * minibatchSize + j)];
			}
		}

		// Build Logistic Regression model
		LogisticLayer classifier = new LogisticLayer(nIn, patterns); // construct logistic regression

		// train
		for (int epoch = 0; epoch < epochs; epoch++) {
			for (int batch = 0; batch < minibatchNumber; batch++) {
				classifier.train(train_X_minibatch[batch], train_T_minibatch[batch], minibatchSize, learningRate);
			}
			learningRate *= 0.95;
		}

		// test
		for (int i = 0; i < testSetSize; i++) predictLabel[i] = classifier.predict(testSet[i]);

		// output
		for (int i = 0; i < testSetSize; i++) {
			System.out.print("[" + testSet[i][0] + ", " + testSet[i][1] + "] -> Prediction: ");
			if (predictLabel[i][0] > predictLabel[i][1]) {
				System.out.print("Positive, ");
				System.out.print("probability = " + predictLabel[i][0]);
			} else {
				System.out.print("Negative, ");
				System.out.print("probability = " + predictLabel[i][1]);
			}
			System.out.print("; Actual: ");
			if (testLabel[i][0] == 1) { System.out.println("Positive"); }
			else { System.out.println("Negative"); }
		}
	}

}
