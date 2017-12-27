package com.isaac.examples;

import com.isaac.initialization.Activation;
import com.isaac.neuralnetworks.DropoutNetworks;
import com.isaac.utils.Evaluation;
import com.isaac.utils.RandomGenerator;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

@SuppressWarnings("Duplicates")
public class DropoutExample {

	public static void main(String[] args) {
		final Random rng = new Random(1234);
        // Declare variables and constants
        int trainSetsSizeEach = 300;
        int testSetsSizeEach = 50;
        int nInEach = 20;
        double pNoiseTrain = 0.2;
        double pNoiseTest = 0.25;

        final int patterns = 3; // nOut
        final int trainSetsSize = trainSetsSizeEach * patterns;
        final int testSetsSize = testSetsSizeEach * patterns;
        final int nIn = nInEach * patterns;

        int[] hiddenLayerSizes = {100, 80};
        double pDropout = 0.5;
        double[][] trainSets = new double[trainSetsSize][nIn];
        int[][] trainLabels = new int[trainSetsSize][patterns];
        double[][] testSets = new double[testSetsSize][nIn];
        Integer[][] testLabels = new Integer[testSetsSize][patterns];
        Integer[][] predictLabels = new Integer[testSetsSize][patterns];

        int epochs = 5000;
        double learningRate = 0.2;
        int minibatchSize = 50;
        final int minibatchNumber = trainSetsSize / minibatchSize;

        double[][][] trainSetMinibatch = new double[minibatchNumber][minibatchSize][nIn];
        int[][][] trainLabelMinibatch = new int[minibatchNumber][minibatchSize][patterns];
        List<Integer> minibatchIndex = new ArrayList<>();
        for (int i = 0; i < trainSetsSize; i++) minibatchIndex.add(i);
        Collections.shuffle(minibatchIndex, rng);

        // Create training data and test data for demo.
        for (int pattern = 0; pattern < patterns; pattern++) {
            for (int n = 0; n < trainSetsSizeEach; n++) {
                int n_ = pattern * trainSetsSizeEach + n;
                for (int i = 0; i < nIn; i++) {
                    if ( (n_ >= trainSetsSizeEach * pattern && n_ < trainSetsSizeEach * (pattern + 1) ) &&
                            (i >= nInEach * pattern && i < nInEach * (pattern + 1)) ) {
                        trainSets[n_][i] = RandomGenerator.binomial(1, 1 - pNoiseTrain, rng) * rng.nextDouble() * .5 + .5;
                    } else {
                        trainSets[n_][i] = RandomGenerator.binomial(1, pNoiseTrain, rng) * rng.nextDouble() * .5 + .5;
                    }
                }
                for (int i = 0; i < patterns; i++) {
                    if (i == pattern) { trainLabels[n_][i] = 1; }
                    else { trainLabels[n_][i] = 0; }
                }
            }

            for (int n = 0; n < testSetsSizeEach; n++) {
                int n_ = pattern * testSetsSizeEach + n;
                for (int i = 0; i < nIn; i++) {
                    if ( (n_ >= testSetsSizeEach * pattern && n_ < testSetsSizeEach * (pattern + 1) ) &&
                            (i >= nInEach * pattern && i < nInEach * (pattern + 1)) ) {
                        testSets[n_][i] = (double) RandomGenerator.binomial(1, 1 - pNoiseTest, rng) *  rng.nextDouble() * .5 + .5;
                    } else {
                        testSets[n_][i] = (double) RandomGenerator.binomial(1, pNoiseTest, rng) *  rng.nextDouble() * .5 + .5;
                    }
                }
                for (int i = 0; i < patterns; i++) {
                    if (i == pattern) { testLabels[n_][i] = 1; }
                    else { testLabels[n_][i] = 0; }
                }
            }
        }

        // create minibatches
        for (int j = 0; j < minibatchSize; j++) {
            for (int i = 0; i < minibatchNumber; i++) {
                trainSetMinibatch[i][j] = trainSets[minibatchIndex.get(i * minibatchSize + j)];
                trainLabelMinibatch[i][j] = trainLabels[minibatchIndex.get(i * minibatchSize + j)];
            }
        }
        // Build Dropout model
        System.out.print("Building the model...");
        DropoutNetworks classifier = new DropoutNetworks(nIn, hiddenLayerSizes, patterns, rng, Activation.ReLU);
        System.out.println("done.");

        // train the model
        System.out.print("Training the model...");
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int batch = 0; batch < minibatchNumber; batch++) {
                classifier.train(trainSetMinibatch[batch], trainLabelMinibatch[batch], minibatchSize, learningRate, pDropout);
            }
            learningRate *= 0.999;
        }
        System.out.println("done.");

        // adjust the weight for testing
        System.out.print("Optimizing weights before testing...");
        classifier.pretest(pDropout);
        System.out.println("done.");

        // test
        for (int i = 0; i < testSetsSize; i++) predictLabels[i] = classifier.predict(testSets[i]);

        // Evaluate the model
        Evaluation evaluation = new Evaluation(predictLabels, testLabels).fit();
        double accuracy = evaluation.getAccuracy();
        double[] precision = evaluation.getPrecision();
        double[] recall = evaluation.getRecall();

        System.out.println("Dropout model evaluation");
        System.out.println("------------------------");
        System.out.printf("Accuracy: %.1f %%\n", accuracy * 100);
        System.out.println("Precision:");
        for (int i = 0; i < patterns; i++) System.out.printf(" class %d: %.1f %%\n", i+1, precision[i] * 100);
        System.out.println("Recall:");
        for (int i = 0; i < patterns; i++) System.out.printf(" class %d: %.1f %%\n", i+1, recall[i] * 100);
	}
}