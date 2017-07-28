package com.isaac.nns.examples.nd4j;

import com.isaac.nns.initialization.Activation;
import com.isaac.nns.initialization.Weight;
import com.isaac.nns.layers.nd4j.OutputLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class LogisticRegresXORExample {
	public static void main(String[] args) {
		final Random rng = new Random(1234); // seed random
		/*
		 * Declare variables and constants
		 */
		final int patterns = 2; // number of classes
		final int train_N = 4;
		final int test_N = 4;
		final int nIn = 2;
		final int nOut = patterns;

		/*
		 * Training data for demo
		 * class 1 : [0, 0], [1, 1] for negative class
		 * class 2 : [0, 1], [1, 0] for positive class
		 */
		INDArray train_X = Nd4j.create(new double[] {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0}, new int[] {train_N, nIn});
		INDArray train_T = Nd4j.create(new double[] {0, 1, 1, 0, 1, 0, 0, 1}, new int[] {train_N, nOut});
		INDArray test_X = Nd4j.create(new double[] {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0}, new int[] {test_N, nIn});
		INDArray test_T = Nd4j.create(new double[] {0, 1, 1, 0, 1, 0, 0, 1}, new int[] {test_N, nOut});

		final int epochs = 2000;
		double learningRate = 0.2;
		int minibatchSize = 1; //  set 1 for on-line training
		int minibatch_N = train_N / minibatchSize;

		List<INDArray> train_X_minibatch = new ArrayList<>();
		List<INDArray> train_T_minibatch = new ArrayList<>();
		List<Integer> minibatchIndex = new ArrayList<>();
		for (int i = 0; i < train_N; i++)
			minibatchIndex.add(i);
		Collections.shuffle(minibatchIndex, rng);

		// create minibatches
		for (int i = 0; i < minibatch_N; i++) {
			INDArray tmpX = Nd4j.create(new double[minibatchSize * nIn], new int[] {minibatchSize, nIn});
			INDArray tmpT = Nd4j.create(new double[minibatchSize * nOut], new int[] {minibatchSize, nOut});
			for (int j = 0; j < minibatchSize; j++) {
				tmpX.putRow(j, train_X.getRow(minibatchIndex.get(i * minibatchSize + j)));
				tmpT.putRow(j, train_T.getRow(minibatchIndex.get(i * minibatchSize + j)));
			}
			train_X_minibatch.add(tmpX);
			train_T_minibatch.add(tmpT);
		}

		/*
		 * Build Logistic Regression model
		 */
		OutputLayer classifier = new OutputLayer(nIn, nOut, Weight.ZERO, Activation.SoftMax, null);

		// train
		for (int epoch = 0; epoch < epochs; epoch++) {
			for (int batch = 0; batch < minibatch_N; batch++) {
				classifier.train(train_X_minibatch.get(batch), train_T_minibatch.get(batch), minibatchSize, learningRate);
			}
			learningRate *= 0.95;
		}

		// test
		INDArray predicted_T = classifier.predict(test_X);

		// evaluate
		for (int i = 0; i < test_N; i++) {
			System.out.print("[" + test_X.getDouble(i, 0) + ", " + test_X.getDouble(i, 1) + "] -> Prediction: ");
			if (predicted_T.getDouble(i, 0) > predicted_T.getDouble(i, 1)) {
				System.out.print("Positive, ");
				System.out.print("probability = " + predicted_T.getDouble(i, 0));
			} else {
				System.out.print("Negative, ");
				System.out.print("probability = " + predicted_T.getDouble(i, 1));
			}
			System.out.print("; Actual: ");
			if (test_T.getDouble(i, 0) == 1) {
				System.out.println("Positive");
			} else {
				System.out.println("Negative");
			}
		}
	}
}
/*
[0.0, 0.0] -> Prediction: Positive, probability = 1.0; Actual: Negative
[0.0, 1.0] -> Prediction: Negative, probability = 1.0; Actual: Positive
[1.0, 0.0] -> Prediction: Positive, probability = 1.0; Actual: Positive
[1.0, 1.0] -> Prediction: Negative, probability = 1.0; Actual: Negative
 */