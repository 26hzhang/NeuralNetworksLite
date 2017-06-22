package com.isaac.java.examples;

import com.isaac.java.neuralnetworks.RestrictedBoltzmannMachine;
import com.isaac.java.utils.RandomGenerator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class RestrictedBoltzmannMachineExample {

	public static void main(String[] args) {
		final Random rng = new Random(123); // seed random
		/**
		 * declare variables and constants
		 */
		int train_N_each = 200;
        int test_N_each = 2;
        int nVisible_each = 4;
        double pNoise_Training = 0.05; // train data noise
        double pNoise_Test = 0.25; // test data noise
        int k = 1; // k times sampling of contrastive divergence
        
        final int patterns = 3;
        final int train_N = train_N_each * patterns;
        final int test_N = test_N_each * patterns;
        final int nVisible = nVisible_each * patterns;
        int nHidden = 6;

        int[][] train_X = new int[train_N][nVisible];
        int[][] test_X = new int[test_N][nVisible];
        double[][] reconstructed_X = new double[test_N][nVisible];

        int epochs = 1000;
        double learningRate = 0.2;
        int minibatchSize = 10;
        final int minibatch_N = train_N / minibatchSize;

        int[][][] train_X_minibatch = new int[minibatch_N][minibatchSize][nVisible];
        List<Integer> minibatchIndex = new ArrayList<Integer>();
        for (int i = 0; i < train_N; i++) minibatchIndex.add(i);
        Collections.shuffle(minibatchIndex, rng); // shuffle data index

        /**
         * Create training data and test data for demo.
         * Data without noise would be:
         * 		class 1 : [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
         * 		class 2 : [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
         * 		class 3 : [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
         * and to each data, we add some noise.
         * For example, one of the data in class 1 could be:
         * 		[1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1]
         */
        for (int pattern = 0; pattern < patterns; pattern++) {
            for (int n = 0; n < train_N_each; n++) {
                int n_ = pattern * train_N_each + n;
                for (int i = 0; i < nVisible; i++) {
                    if ( (n_ >= train_N_each * pattern && n_ < train_N_each * (pattern + 1) ) &&
                            (i >= nVisible_each * pattern && i < nVisible_each * (pattern + 1)) ) {
                        train_X[n_][i] = RandomGenerator.binomial(1, 1-pNoise_Training, rng);
                    } else {
                        train_X[n_][i] = RandomGenerator.binomial(1, pNoise_Training, rng);
                    }
                }
            }
            for (int n = 0; n < test_N_each; n++) {
                int n_ = pattern * test_N_each + n;
                for (int i = 0; i < nVisible; i++) {
                    if ((n_ >= test_N_each * pattern && n_ < test_N_each * (pattern + 1)) &&
                            (i >= nVisible_each * pattern && i < nVisible_each * (pattern + 1))) {
                        test_X[n_][i] = RandomGenerator.binomial(1, 1-pNoise_Test, rng);
                    } else {
                        test_X[n_][i] = RandomGenerator.binomial(1, pNoise_Test, rng);
                    }
                }
            }
        }
        // create minibatches
        for (int i = 0; i < minibatch_N; i++) {
            for (int j = 0; j < minibatchSize; j++) {
                train_X_minibatch[i][j] = train_X[minibatchIndex.get(i * minibatchSize + j)];
            }
        }
        
        /**
         * Build Restricted Boltzmann Machines model
         */
        RestrictedBoltzmannMachine nn = new RestrictedBoltzmannMachine(nVisible, nHidden, null, null, null, rng);
        
        // train with contrastive divergence
        for (int epoch = 0; epoch < epochs; epoch++) {
        	for (int batch = 0; batch < minibatch_N; batch++) {
        		nn.contrastiveDivergence(train_X_minibatch[batch], minibatchSize, learningRate, k);
        	}
        	learningRate *= 0.995; // update learningRate
        }
        
        // test (reconstruct noised data)
        for (int i = 0; i < test_N; i++) {
            reconstructed_X[i] = nn.reconstruct(test_X[i]);
        }
        
        // evaluation
        System.out.println("RBM model reconstruction evaluation");
        for (int pattern = 0; pattern < patterns; pattern++) {
            System.out.printf("Class%d\n", pattern + 1);
            for (int n = 0; n < test_N_each; n++) {
                int n_ = pattern * test_N_each + n;
                System.out.print(Arrays.toString(test_X[n_]) + " -> ");
                System.out.print("[");
                for (int i = 0; i < nVisible-1; i++) {
                    System.out.printf("%.5f, ", reconstructed_X[n_][i]);
                }
                System.out.printf("%.5f]\n", reconstructed_X[n_][nVisible-1]);
            }
            System.out.println();
        }
	}

}
/*
RBM model reconstruction evaluation
-----------------------------------
Class1
[1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0] -> [0.96875, 0.92392, 0.92700, 0.95632, 0.08303, 0.07761, 0.04531, 0.04330, 0.04093, 0.04548, 0.05789, 0.04799]
[1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0] -> [0.94514, 0.88040, 0.86437, 0.93012, 0.17323, 0.14582, 0.09613, 0.08403, 0.02642, 0.02898, 0.03465, 0.02911]

Class2
[1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1] -> [0.06542, 0.07187, 0.08889, 0.06801, 0.89238, 0.87848, 0.85257, 0.88511, 0.09773, 0.12283, 0.10452, 0.06247]
[0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0] -> [0.07713, 0.08592, 0.10869, 0.08299, 0.83253, 0.82172, 0.78342, 0.83456, 0.14273, 0.17267, 0.14879, 0.09271]

Class3
[0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0] -> [0.11811, 0.15057, 0.20546, 0.10047, 0.11103, 0.16121, 0.12962, 0.17031, 0.73920, 0.78998, 0.76806, 0.68131]
[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0] -> [0.03116, 0.05382, 0.05924, 0.01959, 0.06515, 0.10797, 0.10791, 0.09999, 0.86050, 0.90336, 0.88222, 0.83620]
 */