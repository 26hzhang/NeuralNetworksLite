package com.isaac.examples;

import com.isaac.layers.LogisticLayer;
import com.isaac.utils.Evaluation;
import com.isaac.utils.GaussianDistribution;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

@SuppressWarnings("Duplicates")
public class LogisticRegressionExample {

	public static void main(String[] args) {
		final Random rng = new Random(1234); // seed random
		// Declare variables and constants
		final int patterns = 3; // number of classes, nOut
		final int trainSetSize = 400 * patterns;
		final int testLabelSize = 60 * patterns;
		final int nIn = 2;

		double[][] trainSet = new double[trainSetSize][nIn];
		int[][] trainLabel = new int[trainSetSize][patterns];

		double[][] testSet = new double[testLabelSize][nIn];
		Integer[][] testLabel = new Integer[testLabelSize][patterns];
		Integer[][] predictLabel = new Integer[testLabelSize][patterns];

		int epochs = 2000;
		double learningRate = 0.2;

		int minibatchSize = 50; // number of data in each minibatch
		int minibatchNumber = trainSetSize / minibatchSize; // number of minibatches

		double[][][] trainSetMinibatch = new double[minibatchNumber][minibatchSize][nIn]; // minibatches of train data
		int[][][] trainLabelMinibatch = new int[minibatchNumber][minibatchSize][patterns]; // minibatches of output data for training
		List<Integer> minibatchIndex = new ArrayList<>(); // data index for minibatch to apply SGD
		for (int i = 0; i < trainSetSize; i++) minibatchIndex.add(i);
		Collections.shuffle(minibatchIndex, rng); // shuffle data index for SGD

		/*
		 * Training data for demo
		 * class 1 : x1 ~ N(-2.0, 1.0), y1 ~ N(+2.0, 1.0)
		 * class 2 : x2 ~ N(+2.0, 1.0), y2 ~ N(-2.0, 1.0)
		 * class 3 : x3 ~ N( 0.0, 1.0), y3 ~ N( 0.0, 1.0)
		 */
		GaussianDistribution g1 = new GaussianDistribution(-2.0, 1.0, rng);
		GaussianDistribution g2 = new GaussianDistribution(2.0, 1.0, rng);
		GaussianDistribution g3 = new GaussianDistribution(0.0, 1.0, rng);

		// data set in class 1
		for (int i = 0; i < trainSetSize / patterns - 1; i++) {
			trainSet[i][0] = g1.random();
			trainSet[i][1] = g2.random();
			trainLabel[i] = new int[] { 1, 0, 0 };
		}
		for (int i = 0; i < testLabelSize / patterns - 1; i++) {
			testSet[i][0] = g1.random();
			testSet[i][1] = g2.random();
			testLabel[i] = new Integer[] { 1, 0, 0 };
		}

		// data set in class 2
		for (int i = trainSetSize / patterns - 1; i < trainSetSize / patterns * 2 - 1; i++) {
			trainSet[i][0] = g2.random();
			trainSet[i][1] = g1.random();
			trainLabel[i] = new int[] { 0, 1, 0 };
		}
		for (int i = testLabelSize / patterns - 1; i < testLabelSize / patterns * 2 - 1; i++) {
			testSet[i][0] = g2.random();
			testSet[i][1] = g1.random();
			testLabel[i] = new Integer[] { 0, 1, 0 };
		}

		// data set in class 3
		for (int i = trainSetSize / patterns * 2 - 1; i < trainSetSize; i++) {
			trainSet[i][0] = g3.random();
			trainSet[i][1] = g3.random();
			trainLabel[i] = new int[] { 0, 0, 1 };
		}
		for (int i = testLabelSize / patterns * 2 - 1; i < testLabelSize; i++) {
			testSet[i][0] = g3.random();
			testSet[i][1] = g3.random();
			testLabel[i] = new Integer[] { 0, 0, 1 };
		}

		// create minibatches with training data
		for (int i = 0; i < minibatchNumber; i++) {
			for (int j = 0; j < minibatchSize; j++) {
				trainSetMinibatch[i][j] = trainSet[minibatchIndex.get(i * minibatchSize + j)];
				trainLabelMinibatch[i][j] = trainLabel[minibatchIndex.get(i * minibatchSize + j)];
			}
		}

		// Build Logistic Regression model
		LogisticLayer classifier = new LogisticLayer(nIn, patterns); // construct logistic regression

		// train
		for (int epoch = 0; epoch < epochs; epoch++) {
			for (int batch = 0; batch < minibatchNumber; batch++) {
				classifier.train(trainSetMinibatch[batch], trainLabelMinibatch[batch], minibatchSize, learningRate);
			}
			learningRate *= 0.95;
		}

		// test
		for (int i = 0; i < testLabelSize; i++) predictLabel[i] = classifier.predict(testSet[i]);

		//Evaluate the model
		Evaluation evaluation = new Evaluation(predictLabel, testLabel).fit();
		double accuracy = evaluation.getAccuracy();
		double[] precision = evaluation.getPrecision();
		double[] recall = evaluation.getRecall();

		System.out.println("Logistic Regression model evaluation");
		System.out.println(String.format("Accuracy: %.1f\n", accuracy * 100));
		System.out.println("Precision:");
		for (int i = 0; i < patterns; i++) System.out.printf("class %d: %.1f\n", i + 1, precision[i] * 100);
		System.out.println("Recall:");
		for (int i = 0; i < patterns; i++) System.out.printf("class %d: %.1f\n", i + 1, recall[i] * 100);
	}

}