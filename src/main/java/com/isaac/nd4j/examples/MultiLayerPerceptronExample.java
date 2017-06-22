package com.isaac.nd4j.examples;

import com.isaac.nd4j.neuralnetworks.MultiLayerPerceptron;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
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
		final int nOut = patterns; // patterns

		INDArray train_X = Nd4j.create(new double[]{0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0}, new int[]{4, 2});
		INDArray train_T = Nd4j.create(new double[]{0, 1, 1, 0, 1, 0, 0, 1}, new int[]{4, 2});

		INDArray test_X = Nd4j.create(new double[]{0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0}, new int[]{4, 2});
		INDArray test_T = Nd4j.create(new double[]{0, 1, 1, 0, 1, 0, 0, 1}, new int[]{4, 2});

		final int epochs = 5000;
		double learningRate = 0.1;

		final int minibatchSize = 1; //  here, we do online training
		int minibatch_N = train_N / minibatchSize;

		List<INDArray> train_X_minibatch = new ArrayList<>();
		List<INDArray> train_T_minibatch = new ArrayList<>();

		List<Integer> minibatchIndex = new ArrayList<>(); // data index for minibatch to apply SGD
		for (int i = 0; i < train_N; i++) {
			minibatchIndex.add(i);
		}
		Collections.shuffle(minibatchIndex, rng); // shuffle data index for SGD

		// create minibatches with training data
		for (int i = 0; i < minibatch_N; i++) {
			INDArray trainX = Nd4j.create(new double[minibatchSize * nIn], new int[] {minibatchSize, nIn});
			INDArray trainT = Nd4j.create(new double[minibatchSize * nOut], new int[] {minibatchSize, nOut});
			for (int j = 0; j < minibatchSize; j++) {
				trainX.putRow(j, train_X.getRow(minibatchIndex.get(i * minibatchSize + j)));
				trainT.putRow(j, train_T.getRow(minibatchIndex.get(i * minibatchSize + j)));
			}
			train_X_minibatch.add(trainX);
			train_T_minibatch.add(trainT);
		}

		// Build Multi-Layer Perceptrons model
		// construct
		MultiLayerPerceptron classifier = new MultiLayerPerceptron(nIn, nHidden, nOut, rng);

		// train
		for (int epoch = 0; epoch < epochs; epoch++) {
			for (int batch = 0; batch < minibatch_N; batch++) {
				classifier.train(train_X_minibatch.get(batch), train_T_minibatch.get(batch), minibatchSize, learningRate);
			}
		}

		// test
		INDArray predicted_T = classifier.predict(test_X);

		//
		// Evaluate the model
		//
		int[][] confusionMatrix = new int[patterns][patterns];
		double accuracy = 0.;
		double[] precision = new double[patterns];
		double[] recall = new double[patterns];

		for (int i = 0; i < test_N; i++) {
			int predicted_ = 0;
			int actual_ = 0;
			for (int j = 0; j < nOut; j++) {
				if (predicted_T.getDouble(i, j) == 1.0)
					predicted_ = j;
			}
			for (int j = 0; j < nOut; j++) {
				if (test_T.getDouble(i, j) == 1.0)
					actual_ = j;
			}
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