package com.isaac.examples;

import com.isaac.initialization.Activation;
import com.isaac.initialization.WeightInit;
import com.isaac.layers.OutputLayer;
import com.isaac.utils.Evaluation;
import com.isaac.utils.GaussianDistribution;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

@SuppressWarnings("Duplicates")
public class LogisticRegressionExample {

	public static void main(String[] args) {
		final Random rng = new Random(1234); // seed random
		// Declare variables and constants
		final int patterns = 3; // number of classes
		final int trainSetSize = 400 * patterns;
		final int testSetSize = 60 * patterns;
		final int nIn = 2;
		final int nOut = 3;

		INDArray trainSet = Nd4j.create(new double[trainSetSize * nIn], new int[] {trainSetSize, nIn});
		INDArray trainLabel = Nd4j.create(new double[trainSetSize * nOut], new int[] {trainSetSize, nOut});

		INDArray testSet = Nd4j.create(new double[testSetSize * nIn], new int[] {testSetSize, nIn});
		INDArray testLabel = Nd4j.create(new double[testSetSize * nOut], new int[] {testSetSize, nOut});

		int epochs = 2000; /* iteration times */
		double learningRate = 0.2;

		int minibatchSize = 50; // number of data in each minibatch
		int minibatchNumber = trainSetSize / minibatchSize; // number of minibatches

		List<INDArray> trainSetMinibatch = new ArrayList<>();
		List<INDArray> trainLabelMinibatch = new ArrayList<>();

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
			trainSet.put(i, 0, Nd4j.scalar(g1.random()));
			trainSet.put(i, 1, Nd4j.scalar(g2.random()));
			trainLabel.putRow(i, Nd4j.create(new double[] {1.0, 0.0, 0.0}));
		}
		for (int i = 0; i < testSetSize / patterns - 1; i++) {
			testSet.put(i, 0, Nd4j.scalar(g1.random()));
			testSet.put(i, 1, Nd4j.scalar(g2.random()));
			testLabel.putRow(i, Nd4j.create(new double[] {1.0, 0.0, 0.0}));
		}

		// data set in class 2
		for (int i = trainSetSize / patterns - 1; i < trainSetSize / patterns * 2 - 1; i++) {
			trainSet.put(i, 0, Nd4j.scalar(g2.random()));
			trainSet.put(i, 1, Nd4j.scalar(g1.random()));
			trainLabel.putRow(i, Nd4j.create(new double[] {0.0, 1.0, 0.0}));
		}
		for (int i = testSetSize / patterns - 1; i < testSetSize / patterns * 2 - 1; i++) {
			testSet.put(i, 0, Nd4j.scalar(g2.random()));
			testSet.put(i, 1, Nd4j.scalar(g1.random()));
			testLabel.putRow(i, Nd4j.create(new double[] {0.0, 1.0, 0.0}));
		}

		// data set in class 3
		for (int i = trainSetSize / patterns * 2 - 1; i < trainSetSize; i++) {
			trainSet.put(i, 0, Nd4j.scalar(g3.random()));
			trainSet.put(i, 1, Nd4j.scalar(g3.random()));
			trainLabel.putRow(i, Nd4j.create(new double[] {0.0, 0.0, 1.0}));
		}
		for (int i = testSetSize / patterns * 2 - 1; i < testSetSize; i++) {
			testSet.put(i, 0, Nd4j.scalar(g3.random()));
			testSet.put(i, 1, Nd4j.scalar(g3.random()));
			testLabel.putRow(i, Nd4j.create(new double[] {0.0, 0.0, 1.0}));
		}

		// create minibatches with training data
		for (int i = 0; i < minibatchNumber; i++) {
			INDArray trainX = Nd4j.create(new double[minibatchSize * nIn], new int[] {minibatchSize, nIn});
			INDArray trainT = Nd4j.create(new double[minibatchSize * nOut], new int[] {minibatchSize, nOut});
			for (int j = 0; j < minibatchSize; j++) {
				trainX.putRow(j, trainSet.getRow(minibatchIndex.get(i * minibatchSize + j)));
				trainT.putRow(j, trainLabel.getRow(minibatchIndex.get(i * minibatchSize + j)));
			}
			trainSetMinibatch.add(trainX);
			trainLabelMinibatch.add(trainT);
		}

		// Build Logistic Regression model
		OutputLayer classifier = new OutputLayer(nIn, nOut, WeightInit.ZERO, null, Activation.Softmax);

		// train
		for (int epoch = 0; epoch < epochs; epoch++) {
			for (int batch = 0; batch < minibatchNumber; batch++) {
				classifier.train(trainSetMinibatch.get(batch), trainLabelMinibatch.get(batch), minibatchSize,
						learningRate);
			}
			learningRate *= 0.95;
		}
		// test
		INDArray predicted_T = classifier.predict(testSet);

		// Evaluate the model
		Evaluation evaluation = new Evaluation(predicted_T, testLabel).fit();
		double accuracy = evaluation.getAccuracy();
		double[] precision = evaluation.getPrecision();
		double[] recall = evaluation.getRecall();

		System.out.println("Logistic Regression model evaluation");
		System.out.println("------------------------------------");
		System.out.println(String.format("Accuracy: %.1f\n", accuracy * 100));
		System.out.println("Precision:");
		for (int i = 0; i < patterns; i++) {
			System.out.printf("class %d: %.1f\n", i + 1, precision[i] * 100);
		}
		System.out.println("Recall:");
		for (int i = 0; i < patterns; i++) {
			System.out.printf("class %d: %.1f\n", i + 1, recall[i] * 100);
		}
	}

}