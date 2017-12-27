package com.isaac.examples;

import com.isaac.neuralnetworks.MultiLayerPerceptron;
import com.isaac.utils.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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

		INDArray trainSet = Nd4j.create(new double[]{0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0}, new int[]{trainSetSize, 2});
		INDArray trainLabel = Nd4j.create(new double[]{0, 1, 1, 0, 1, 0, 0, 1}, new int[]{4, 2});

		INDArray testSet = Nd4j.create(new double[]{0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0}, new int[]{testSetSize, 2});
		INDArray testLabel = Nd4j.create(new double[]{0, 1, 1, 0, 1, 0, 0, 1}, new int[]{4, 2});

		final int epochs = 5000;
		double learningRate = 0.1;

		final int minibatchSize = 1; //  here, we do online training
		int minibatchNumber = trainSetSize / minibatchSize;

		List<INDArray> trainSetMinibatch = new ArrayList<>();
		List<INDArray> trainLabelMinibatch = new ArrayList<>();

		List<Integer> minibatchIndex = new ArrayList<>(); // data index for minibatch to apply SGD
		for (int i = 0; i < trainSetSize; i++) minibatchIndex.add(i);
		Collections.shuffle(minibatchIndex, rng); // shuffle data index for SGD

		// create minibatches with training data
		for (int i = 0; i < minibatchNumber; i++) {
			INDArray trainX = Nd4j.create(new double[minibatchSize * nIn], new int[] {minibatchSize, nIn});
			INDArray trainT = Nd4j.create(new double[minibatchSize * patterns], new int[] {minibatchSize, patterns});
			for (int j = 0; j < minibatchSize; j++) {
				trainX.putRow(j, trainSet.getRow(minibatchIndex.get(i * minibatchSize + j)));
				trainT.putRow(j, trainLabel.getRow(minibatchIndex.get(i * minibatchSize + j)));
			}
			trainSetMinibatch.add(trainX);
			trainLabelMinibatch.add(trainT);
		}

		// Build Multi-Layer Perceptrons model
		// construct
		MultiLayerPerceptron classifier = new MultiLayerPerceptron(nIn, nHidden, patterns, rng);

		// train
		for (int epoch = 0; epoch < epochs; epoch++) {
			for (int batch = 0; batch < minibatchNumber; batch++) {
				classifier.train(trainSetMinibatch.get(batch), trainLabelMinibatch.get(batch), minibatchSize, learningRate);
			}
		}

		// test
		INDArray predicted_T = classifier.predict(testSet);

		// Evaluate the model
		Evaluation evaluation = new Evaluation(predicted_T, testLabel).fit();
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