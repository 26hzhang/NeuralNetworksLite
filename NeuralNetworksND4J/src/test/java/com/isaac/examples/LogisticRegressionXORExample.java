package com.isaac.examples;

import com.isaac.initialization.Activation;
import com.isaac.initialization.WeightInit;
import com.isaac.layers.OutputLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

@SuppressWarnings("Duplicates")
public class LogisticRegressionXORExample {
	public static void main(String[] args) {
		final Random rng = new Random(1234); // seed random
		// Declare variables and constants
		final int patterns = 2; // number of classes, nOut
		final int trainSetSize = 4;
		final int testSetSize = 4;
		final int nIn = 2;

		/*
		 * Training data for demo
		 * class 1 : [0, 0], [1, 1] for negative class
		 * class 2 : [0, 1], [1, 0] for positive class
		 */
		INDArray trainSet = Nd4j.create(new double[] {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0}, new int[] {trainSetSize, nIn});
		INDArray trainLabel = Nd4j.create(new double[] {0, 1, 1, 0, 1, 0, 0, 1}, new int[] {trainSetSize, patterns});
		INDArray testSet = Nd4j.create(new double[] {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0}, new int[] {testSetSize, nIn});
		INDArray testLabel = Nd4j.create(new double[] {0, 1, 1, 0, 1, 0, 0, 1}, new int[] {testSetSize, patterns});

		final int epochs = 2000;
		double learningRate = 0.2;
		int minibatchSize = 1; //  set 1 for on-line training
		int minibatchNumber = trainSetSize / minibatchSize;

		List<INDArray> train_X_minibatch = new ArrayList<>();
		List<INDArray> train_T_minibatch = new ArrayList<>();
		List<Integer> minibatchIndex = new ArrayList<>();
		for (int i = 0; i < trainSetSize; i++) minibatchIndex.add(i);
		Collections.shuffle(minibatchIndex, rng);

		// create minibatches
		for (int i = 0; i < minibatchNumber; i++) {
			INDArray tmpX = Nd4j.create(new double[minibatchSize * nIn], new int[] {minibatchSize, nIn});
			INDArray tmpT = Nd4j.create(new double[minibatchSize * patterns], new int[] {minibatchSize, patterns});
			for (int j = 0; j < minibatchSize; j++) {
				tmpX.putRow(j, trainSet.getRow(minibatchIndex.get(i * minibatchSize + j)));
				tmpT.putRow(j, trainLabel.getRow(minibatchIndex.get(i * minibatchSize + j)));
			}
			train_X_minibatch.add(tmpX);
			train_T_minibatch.add(tmpT);
		}

		// Build Logistic Regression model
		OutputLayer classifier = new OutputLayer(nIn, patterns, WeightInit.ZERO, null, Activation.Softmax);

		// train
		for (int epoch = 0; epoch < epochs; epoch++) {
			for (int batch = 0; batch < minibatchNumber; batch++) {
				classifier.train(train_X_minibatch.get(batch), train_T_minibatch.get(batch), minibatchSize, learningRate);
			}
			learningRate *= 0.95;
		}

		// test
		INDArray predicted_T = classifier.predict(testSet);

		// evaluate
		for (int i = 0; i < testSetSize; i++) {
			System.out.print("[" + testSet.getDouble(i, 0) + ", " + testSet.getDouble(i, 1) + "] -> Prediction: ");
			if (predicted_T.getDouble(i, 0) > predicted_T.getDouble(i, 1)) {
				System.out.print("Positive, ");
				System.out.print("probability = " + predicted_T.getDouble(i, 0));
			} else {
				System.out.print("Negative, ");
				System.out.print("probability = " + predicted_T.getDouble(i, 1));
			}
			System.out.print("; Actual: ");
			if (testLabel.getDouble(i, 0) == 1) {
				System.out.println("Positive");
			} else {
				System.out.println("Negative");
			}
		}
	}
}