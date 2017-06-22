package com.isaac.java.examples;

import com.isaac.java.neuralnetworks.MultiLayerPerceptron;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class MultiLayerPerceptronExample {

	public static void main(String[] args) {
		final Random rng = new Random(123); // seed random
		//
		// Declare variables and constants
		//
		final int patterns = 2;
		final int train_N = 4;
		final int test_N = 4;
		final int nIn = 2;
		final int nHidden = 3;
		final int nOut = patterns;

		double[][] train_X;
		int[][] train_T;

		double[][] test_X;
		Integer[][] test_T;
		Integer[][] predicted_T = new Integer[test_N][nOut];

		final int epochs = 5000;
		double learningRate = 0.1;

		final int minibatchSize = 1; //  here, we do on-line training
		int minibatch_N = train_N / minibatchSize;

		double[][][] train_X_minibatch = new double[minibatch_N][minibatchSize][nIn];
		int[][][] train_T_minibatch = new int[minibatch_N][minibatchSize][nOut];
		List<Integer> minibatchIndex = new ArrayList<>();
		for (int i = 0; i < train_N; i++)
			minibatchIndex.add(i);
		Collections.shuffle(minibatchIndex, rng);

		//
		// Training simple XOR problem for demo
		//   class 1 : [0, 0], [1, 1]  ->  Negative [0, 1]
		//   class 2 : [0, 1], [1, 0]  ->  Positive [1, 0]
		//

		train_X = new double[][] { { 0., 0. }, { 0., 1. }, { 1., 0. }, { 1., 1. } };
		train_T = new int[][] { { 0, 1 }, { 1, 0 }, { 1, 0 }, { 0, 1 } };
		test_X = new double[][] { { 0., 0. }, { 0., 1. }, { 1., 0. }, { 1., 1. } };
		test_T = new Integer[][] { { 0, 1 }, { 1, 0 }, { 1, 0 }, { 0, 1 } };

		// create minibatches
		for (int i = 0; i < minibatch_N; i++) {
			for (int j = 0; j < minibatchSize; j++) {
				train_X_minibatch[i][j] = train_X[minibatchIndex.get(i * minibatchSize + j)];
				train_T_minibatch[i][j] = train_T[minibatchIndex.get(i * minibatchSize + j)];
			}
		}

		//
		// Build Multi-Layer Perceptrons model
		//

		// construct
		MultiLayerPerceptron classifier = new MultiLayerPerceptron(nIn, nHidden, nOut, rng);

		// train
		for (int epoch = 0; epoch < epochs; epoch++) {
			for (int batch = 0; batch < minibatch_N; batch++) {
				classifier.train(train_X_minibatch[batch], train_T_minibatch[batch], minibatchSize, learningRate);
			}
		}

		// test
		for (int i = 0; i < test_N; i++) {
			predicted_T[i] = classifier.predict(test_X[i]);
		}

		//
		// Evaluate the model
		//

		int[][] confusionMatrix = new int[patterns][patterns];
		double accuracy = 0.;
		double[] precision = new double[patterns];
		double[] recall = new double[patterns];

		for (int i = 0; i < test_N; i++) {
			int predicted_ = Arrays.asList(predicted_T[i]).indexOf(1);
			int actual_ = Arrays.asList(test_T[i]).indexOf(1);

			confusionMatrix[actual_][predicted_] += 1;
		}

		for (int i = 0; i < patterns; i++) {
			double col_ = 0.;
			double row_ = 0.;

			for (int j = 0; j < patterns; j++) {

				if (i == j) {
					accuracy += confusionMatrix[i][j];
					precision[i] += confusionMatrix[j][i];
					recall[i] += confusionMatrix[i][j];
				}

				col_ += confusionMatrix[j][i];
				row_ += confusionMatrix[i][j];
			}
			precision[i] /= col_;
			recall[i] /= row_;
		}

		accuracy /= test_N;

		System.out.println("--------------------");
		System.out.println("MLP model evaluation");
		System.out.println("--------------------");
		System.out.printf("Accuracy: %.1f %%\n", accuracy * 100);
		System.out.println("Precision:");
		for (int i = 0; i < patterns; i++) {
			System.out.printf(" class %d: %.1f %%\n", i + 1, precision[i] * 100);
		}
		System.out.println("Recall:");
		for (int i = 0; i < patterns; i++) {
			System.out.printf(" class %d: %.1f %%\n", i + 1, recall[i] * 100);
		}
	}

}
/*
MLP model evaluation
--------------------
Accuracy: 100.0 %
Precision:
 class 1: 100.0 %
 class 2: 100.0 %
Recall:
 class 1: 100.0 %
 class 2: 100.0 %
 */