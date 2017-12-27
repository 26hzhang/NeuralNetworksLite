package com.isaac.examples;

import com.isaac.initialization.Activation;
import com.isaac.neuralnetworks.MultiLayerPerceptron;
import com.isaac.utils.Evaluation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class MultiLayerPerceptronExample {

	public static void main(String[] args) {
		final Random rng = new Random(123); // seed random
		// Declare variables and constants
		final int patterns = 2; // nOut
		final int trainSetSize = 4;
		final int testSetSize = 4;
		final int nIn = 2;
		final int nHidden = 3;

		double[][] trainSet;
		int[][] trainLabel;
		double[][] testSet;
		Integer[][] testLabel;
		Integer[][] predictLabel = new Integer[testSetSize][patterns];

		final int epochs = 5000;
		double learningRate = 0.1;

		final int minibatchSize = 1; //  here, we do on-line training
		int minibatchNumber = trainSetSize / minibatchSize;

		double[][][] trainSetMinibatch = new double[minibatchNumber][minibatchSize][nIn];
		int[][][] trainLabelMinibatch = new int[minibatchNumber][minibatchSize][patterns];
		List<Integer> minibatchIndex = new ArrayList<>();
		for (int i = 0; i < trainSetSize; i++) minibatchIndex.add(i);
		Collections.shuffle(minibatchIndex, rng);

		// Training simple XOR problem for demo
		//   class 1 : [0, 0], [1, 1]  ->  Negative [0, 1]
		//   class 2 : [0, 1], [1, 0]  ->  Positive [1, 0]
		trainSet = new double[][] { { 0., 0. }, { 0., 1. }, { 1., 0. }, { 1., 1. } };
		trainLabel = new int[][] { { 0, 1 }, { 1, 0 }, { 1, 0 }, { 0, 1 } };
		testSet = new double[][] { { 0., 0. }, { 0., 1. }, { 1., 0. }, { 1., 1. } };
		testLabel = new Integer[][] { { 0, 1 }, { 1, 0 }, { 1, 0 }, { 0, 1 } };

		// create minibatches
		for (int i = 0; i < minibatchNumber; i++) {
			for (int j = 0; j < minibatchSize; j++) {
				trainSetMinibatch[i][j] = trainSet[minibatchIndex.get(i * minibatchSize + j)];
				trainLabelMinibatch[i][j] = trainLabel[minibatchIndex.get(i * minibatchSize + j)];
			}
		}

		// Build Multi-Layer Perceptrons model
		MultiLayerPerceptron classifier = new MultiLayerPerceptron(nIn, nHidden, patterns, rng, Activation.Tanh);

		// train
		for (int epoch = 0; epoch < epochs; epoch++) {
			for (int batch = 0; batch < minibatchNumber; batch++) {
				classifier.train(trainSetMinibatch[batch], trainLabelMinibatch[batch], minibatchSize, learningRate);
			}
		}

		// test
		for (int i = 0; i < testSetSize; i++) predictLabel[i] = classifier.predict(testSet[i]);

		// Evaluate the model
		Evaluation evaluation = new Evaluation(predictLabel, testLabel).fit();
		double accuracy = evaluation.getAccuracy();
		double[] precision = evaluation.getPrecision();
		double[] recall = evaluation.getRecall();

		System.out.println("MLP model evaluation");
		System.out.println("--------------------");
		System.out.printf("Accuracy: %.1f %%\n", accuracy * 100);
		System.out.println("Precision:");
		for (int i = 0; i < patterns; i++) System.out.printf(" class %d: %.1f %%\n", i + 1, precision[i] * 100);
		System.out.println("Recall:");
		for (int i = 0; i < patterns; i++) System.out.printf(" class %d: %.1f %%\n", i + 1, recall[i] * 100);
	}
}