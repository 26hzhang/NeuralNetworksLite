package com.isaac.examples;

import com.isaac.initialization.Activation;
import com.isaac.layers.DenoisingAutoencoder;
import com.isaac.utils.RandomGenerator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

@SuppressWarnings("Duplicates")
public class DenoisingAutoencoderExample {

	public static void main(String[] args) {
		final Random rng = new Random(1234);
        // Declare variables and constants
        int trainSetSizeEach = 200;
        int testSetSizeEach = 2;
        int nVisibleEach = 4;
        double pNoiseTrain = 0.05;
        double pNoiseTest = 0.25;

        final int patterns = 3;
        final int trainSetSize = trainSetSizeEach * patterns;
        final int testSetSize = testSetSizeEach * patterns;
        final int nVisible = nVisibleEach * patterns;
        int nHidden = 6;
        double corruptionLevel = 0.3;

        double[][] trainSet = new double[trainSetSize][nVisible];
        double[][] testSet = new double[testSetSize][nVisible];
        double[][] reconstructedSet = new double[testSetSize][nVisible];

        int epochs = 1000;
        double learningRate = 0.2;
        int minibatchSize = 10;
        final int minibatchNumber = trainSetSize / minibatchSize;

        double[][][] trainSetMinibatch = new double[minibatchNumber][minibatchSize][nVisible];
        List<Integer> minibatchIndex = new ArrayList<>();
        for (int i = 0; i < trainSetSize; i++) minibatchIndex.add(i);
        Collections.shuffle(minibatchIndex, rng);

        // Create training data and test data for demo.
        //   Data without noise would be:
        //     class 1 : [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        //     class 2 : [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
        //     class 3 : [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        //   and to each data, we add some noise.
        //   For example, one of the data in class 1 could be:
        //     [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1]
        for (int pattern = 0; pattern < patterns; pattern++) {
            for (int n = 0; n < trainSetSizeEach; n++) {
                int n_ = pattern * trainSetSizeEach + n;
                for (int i = 0; i < nVisible; i++) {
                    if ( (n_ >= trainSetSizeEach * pattern && n_ < trainSetSizeEach * (pattern + 1) ) &&
                            (i >= nVisibleEach * pattern && i < nVisibleEach * (pattern + 1)) ) {
                        trainSet[n_][i] = RandomGenerator.binomial(1, 1 - pNoiseTrain, rng);
                    } else { trainSet[n_][i] = RandomGenerator.binomial(1, pNoiseTrain, rng); }
                }
            }
            for (int n = 0; n < testSetSizeEach; n++) {
                int n_ = pattern * testSetSizeEach + n;
                for (int i = 0; i < nVisible; i++) {
                    if ( (n_ >= testSetSizeEach * pattern && n_ < testSetSizeEach * (pattern + 1) ) &&
                            (i >= nVisibleEach * pattern && i < nVisibleEach * (pattern + 1)) ) {
                        testSet[n_][i] = RandomGenerator.binomial(1, 1 - pNoiseTest, rng);
                    } else { testSet[n_][i] = RandomGenerator.binomial(1, pNoiseTest, rng); }
                }
            }
        }

        // create minibatches
        for (int i = 0; i < minibatchNumber; i++) {
            for (int j = 0; j < minibatchSize; j++) {
                trainSetMinibatch[i][j] = trainSet[minibatchIndex.get(i * minibatchSize + j)];
            }
        }

        // Build Denoising AutoEncoder Model
        // construct DA
        DenoisingAutoencoder nn = new DenoisingAutoencoder(nVisible, nHidden, null, null, null, rng,
                Activation.Sigmoid);

        // train
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int batch = 0; batch < minibatchNumber; batch++) {
                nn.train(trainSetMinibatch[batch], minibatchSize, learningRate, corruptionLevel);
            }
        }

        // test (reconstruct noised data)
        for (int i = 0; i < testSetSize; i++) reconstructedSet[i] = nn.reconstruct(testSet[i]);

        // evaluation
        System.out.println("-----------------------------------");
        System.out.println("DA model reconstruction evaluation");
        System.out.println("-----------------------------------");
        for (int pattern = 0; pattern < patterns; pattern++) {
            System.out.printf("Class%d\n", pattern + 1);
            for (int n = 0; n < testSetSizeEach; n++) {
                int n_ = pattern * testSetSizeEach + n;
                System.out.print(Arrays.toString(testSet[n_]) + " -> ");
                System.out.print("[");
                for (int i = 0; i < nVisible-1; i++) {
                    System.out.printf("%.5f, ", reconstructedSet[n_][i]);
                }
                System.out.printf("%.5f]\n", reconstructedSet[n_][nVisible-1]);
            }
            System.out.println();
        }
	}

}