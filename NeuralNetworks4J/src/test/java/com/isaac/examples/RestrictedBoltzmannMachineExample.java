package com.isaac.examples;

import com.isaac.initialization.Activation;
import com.isaac.layers.RestrictedBoltzmannMachine;
import com.isaac.utils.RandomGenerator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

@SuppressWarnings("Duplicates")
public class RestrictedBoltzmannMachineExample {

	public static void main(String[] args) {
		final Random rng = new Random(123); // seed random
		// declare variables and constants
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
        List<Integer> minibatchIndex = new ArrayList<>();
        for (int i = 0; i < train_N; i++) minibatchIndex.add(i);
        Collections.shuffle(minibatchIndex, rng); // shuffle data index

        /*
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
                    } else { train_X[n_][i] = RandomGenerator.binomial(1, pNoise_Training, rng); }
                }
            }
            for (int n = 0; n < test_N_each; n++) {
                int n_ = pattern * test_N_each + n;
                for (int i = 0; i < nVisible; i++) {
                    if ((n_ >= test_N_each * pattern && n_ < test_N_each * (pattern + 1)) &&
                            (i >= nVisible_each * pattern && i < nVisible_each * (pattern + 1))) {
                        test_X[n_][i] = RandomGenerator.binomial(1, 1-pNoise_Test, rng);
                    } else { test_X[n_][i] = RandomGenerator.binomial(1, pNoise_Test, rng); }
                }
            }
        }
        // create minibatches
        for (int i = 0; i < minibatch_N; i++) {
            for (int j = 0; j < minibatchSize; j++) {
                train_X_minibatch[i][j] = train_X[minibatchIndex.get(i * minibatchSize + j)];
            }
        }
        
        // Build Restricted Boltzmann Machines model
        RestrictedBoltzmannMachine nn = new RestrictedBoltzmannMachine(nVisible, nHidden, null, null, null, rng,
                Activation.Sigmoid);
        
        // train with contrastive divergence
        for (int epoch = 0; epoch < epochs; epoch++) {
        	for (int batch = 0; batch < minibatch_N; batch++)
        	    nn.contrastiveDivergence(train_X_minibatch[batch], minibatchSize, learningRate, k);
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
                for (int i = 0; i < nVisible-1; i++) System.out.printf("%.5f, ", reconstructed_X[n_][i]);
                System.out.printf("%.5f]\n", reconstructed_X[n_][nVisible-1]);
            }
            System.out.println();
        }
	}

}