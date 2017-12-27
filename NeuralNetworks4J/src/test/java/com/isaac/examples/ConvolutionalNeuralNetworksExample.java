package com.isaac.examples;

import com.isaac.initialization.Activation;
import com.isaac.neuralnetworks.ConvolutionalNeuralNetworks;
import com.isaac.utils.Evaluation;
import com.isaac.utils.RandomGenerator;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;


public class ConvolutionalNeuralNetworksExample {

	public static void main(String[] args) {
		final Random rng = new Random(1234);  // seed random
        // Declare variables and constants
        int trainSetSizeEach = 50;
        int testSetSizeEach = 10;
        double pNoiseTrain = 0.05;
        double pNoiseTest = 0.10;
        final int patterns = 3; // nOut
        final int trainSetSize = trainSetSizeEach * patterns;
        final int testSetSize = testSetSizeEach * patterns;
        final int[] imageSize = {12, 12};
        final int channel = 1;
        int[] nKernels = {10, 20};
        int[][] kernelSizes = { {3, 3}, {2, 2} };
        int[][] poolSizes = { {2, 2}, {2, 2} };

        int nHidden = 20;

        double[][][][] trainSet = new double[trainSetSize][channel][imageSize[0]][imageSize[1]];
        int[][] trainLabel = new int[trainSetSize][patterns];
        double[][][][] testSet = new double[testSetSize][channel][imageSize[0]][imageSize[1]];
        Integer[][] testLabel = new Integer[testSetSize][patterns];
        Integer[][] predictLabel = new Integer[testSetSize][patterns];

        int epochs = 500;
        double learningRate = 0.1;

        final int minibatchSize = 25;
        int minibatchNumber = trainSetSize / minibatchSize;

        double[][][][][] trainSetMinibatch = new double[minibatchNumber][minibatchSize][channel][imageSize[0]][imageSize[1]];
        int[][][] trainLabelMinibatch = new int[minibatchNumber][minibatchSize][patterns];
        List<Integer> minibatchIndex = new ArrayList<>();
        for (int i = 0; i < trainSetSize; i++) minibatchIndex.add(i);
        Collections.shuffle(minibatchIndex, rng);

        // Create training data and test data for demo.
        for (int pattern = 0; pattern < patterns; pattern++) {
            for (int n = 0; n < trainSetSizeEach; n++) {
                int n_ = pattern * trainSetSizeEach + n;
                for (int c = 0; c < channel; c++) {
                    for (int i = 0; i < imageSize[0]; i++) {
                        for (int j = 0; j < imageSize[1]; j++) {
                            if ((i < (pattern + 1) * (imageSize[0] / patterns)) && (i >= pattern * imageSize[0] / patterns)) {
                                trainSet[n_][c][i][j] = ((int) 128. * rng.nextDouble() + 128.) * RandomGenerator.binomial(1, 1 - pNoiseTrain, rng) / 256.;
                            } else {
                                trainSet[n_][c][i][j] = 128. * RandomGenerator.binomial(1, pNoiseTrain, rng) / 256.;
                            }
                        }
                    }
                }
                for (int i = 0; i < patterns; i++) {
                    if (i == pattern) { trainLabel[n_][i] = 1; }
                    else { trainLabel[n_][i] = 0; }
                }
            }
            for (int n = 0; n < testSetSizeEach; n++) {
                int n_ = pattern * testSetSizeEach + n;
                for (int c = 0; c < channel; c++) {
                    for (int i = 0; i < imageSize[0]; i++) {
                        for (int j = 0; j < imageSize[1]; j++) {
                            if ((i < (pattern + 1) * imageSize[0] / patterns) && (i >= pattern * imageSize[0] / patterns)) {
                                testSet[n_][c][i][j] = ((int) 128. * rng.nextDouble() + 128.) * RandomGenerator.binomial(1, 1 - pNoiseTest, rng) / 256.;
                            } else {
                                testSet[n_][c][i][j] = 128. * RandomGenerator.binomial(1, pNoiseTest, rng) / 256.;
                            }
                        }
                    }
                }
                for (int i = 0; i < patterns; i++) {
                    if (i == pattern) { testLabel[n_][i] = 1; }
                    else { testLabel[n_][i] = 0; }
                }
            }
        }

        // create minibatches
        for (int j = 0; j < minibatchSize; j++) {
            for (int i = 0; i < minibatchNumber; i++) {
                trainSetMinibatch[i][j] = trainSet[minibatchIndex.get(i * minibatchSize + j)];
                trainLabelMinibatch[i][j] = trainLabel[minibatchIndex.get(i * minibatchSize + j)];
            }
        }

        // Build Convolutional Neural Networks model
        System.out.print("Building the model...");
        ConvolutionalNeuralNetworks classifier = new ConvolutionalNeuralNetworks(imageSize, channel, nKernels, kernelSizes, poolSizes, nHidden, patterns, rng, Activation.ReLU);
        System.out.println("done.");
        // train the model
        System.out.print("Training the model...");
        System.out.println();
        for (int epoch = 0; epoch < epochs; epoch++) {
            if ((epoch + 1) % 50 == 0) System.out.println("\titer = " + (epoch + 1) + " / " + epochs);
            for (int batch = 0; batch < minibatchNumber; batch++) {
                classifier.train(trainSetMinibatch[batch], trainLabelMinibatch[batch], minibatchSize, learningRate);
            }
            learningRate *= 0.999;
        }
        System.out.println("done.");
        // test
        for (int i = 0; i < testSetSize; i++) {
            predictLabel[i] = classifier.predict(testSet[i]);
        }

        // Evaluate the model
        Evaluation evaluation = new Evaluation(predictLabel, testLabel).fit();
        double accuracy = evaluation.getAccuracy();
        double[] precision = evaluation.getPrecision();
        double[] recall = evaluation.getRecall();

        System.out.println("CNN model evaluation");
        System.out.println("--------------------");
        System.out.printf("Accuracy: %.1f %%\n", accuracy * 100);
        System.out.println("Precision:");
        for (int i = 0; i < patterns; i++) System.out.printf(" class %d: %.1f %%\n", i+1, precision[i] * 100);
        System.out.println("Recall:");
        for (int i = 0; i < patterns; i++) System.out.printf(" class %d: %.1f %%\n", i+1, recall[i] * 100);
	}
}