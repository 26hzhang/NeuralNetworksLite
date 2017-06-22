package com.isaac.java.examples;

import com.isaac.java.utils.RandomGenerator;
import com.isaac.java.neuralnetworks.DenoisingAutoencoder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class DenoisingAutoencoderExample {

	public static void main(String[] args) {
		final Random rng = new Random(1234);
        // Declare variables and constants
        int train_N_each = 200;           // for demo
        int test_N_each = 2;              // for demo
        int nVisible_each = 4;           // for demo
        double pNoise_Training = 0.05;     // for demo
        double pNoise_Test = 0.25;         // for demo

        final int patterns = 3;
        final int train_N = train_N_each * patterns;
        final int test_N = test_N_each * patterns;
        final int nVisible = nVisible_each * patterns;
        int nHidden = 6;
        double corruptionLevel = 0.3;

        double[][] train_X = new double[train_N][nVisible];
        double[][] test_X = new double[test_N][nVisible];
        double[][] reconstructed_X = new double[test_N][nVisible];

        int epochs = 1000;
        double learningRate = 0.2;
        int minibatchSize = 10;
        final int minibatch_N = train_N / minibatchSize;

        double[][][] train_X_minibatch = new double[minibatch_N][minibatchSize][nVisible];
        List<Integer> minibatchIndex = new ArrayList<>();
        for (int i = 0; i < train_N; i++) minibatchIndex.add(i);
        Collections.shuffle(minibatchIndex, rng);

        //
        // Create training data and test data for demo.
        //   Data without noise would be:
        //     class 1 : [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        //     class 2 : [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
        //     class 3 : [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        //   and to each data, we add some noise.
        //   For example, one of the data in class 1 could be:
        //     [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1]
        //
        for (int pattern = 0; pattern < patterns; pattern++) {
            for (int n = 0; n < train_N_each; n++) {
                int n_ = pattern * train_N_each + n;
                for (int i = 0; i < nVisible; i++) {
                    if ( (n_ >= train_N_each * pattern && n_ < train_N_each * (pattern + 1) ) &&
                            (i >= nVisible_each * pattern && i < nVisible_each * (pattern + 1)) ) {
                        train_X[n_][i] = RandomGenerator.binomial(1, 1 - pNoise_Training, rng);
                    } else {
                        train_X[n_][i] = RandomGenerator.binomial(1, pNoise_Training, rng);
                    }
                }
            }
            for (int n = 0; n < test_N_each; n++) {
                int n_ = pattern * test_N_each + n;
                for (int i = 0; i < nVisible; i++) {
                    if ( (n_ >= test_N_each * pattern && n_ < test_N_each * (pattern + 1) ) &&
                            (i >= nVisible_each * pattern && i < nVisible_each * (pattern + 1)) ) {
                        test_X[n_][i] = RandomGenerator.binomial(1, 1 - pNoise_Test, rng);
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

        // Build Denoising Autoencoders Model
        // construct DA
        DenoisingAutoencoder nn = new DenoisingAutoencoder(nVisible, nHidden, null, null, null, rng);

        // train
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int batch = 0; batch < minibatch_N; batch++) {
                nn.train(train_X_minibatch[batch], minibatchSize, learningRate, corruptionLevel);
            }
        }

        // test (reconstruct noised data)
        for (int i = 0; i < test_N; i++) {
            reconstructed_X[i] = nn.reconstruct(test_X[i]);
        }

        // evaluation
        System.out.println("-----------------------------------");
        System.out.println("DA model reconstruction evaluation");
        System.out.println("-----------------------------------");
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
DA model reconstruction evaluation
-----------------------------------
Class1
[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0] -> [0.90646, 0.96649, 0.97250, 0.96790, 0.07095, 0.02743, 0.06600, 0.02793, 0.06146, 0.04536, 0.02471, 0.02262]
[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0] -> [0.94949, 0.91284, 0.95034, 0.89624, 0.39246, 0.32690, 0.19030, 0.33091, 0.02301, 0.01095, 0.00306, 0.00204]

Class2
[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] -> [0.22270, 0.08803, 0.23352, 0.38367, 0.97357, 0.98208, 0.93622, 0.92541, 0.00438, 0.00883, 0.00226, 0.86864]
[0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0] -> [0.02752, 0.01565, 0.02460, 0.05879, 0.92471, 0.99871, 0.89367, 0.92562, 0.02566, 0.06760, 0.01280, 0.01905]

Class3
[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0] -> [0.08936, 0.02280, 0.05756, 0.10114, 0.00736, 0.02238, 0.01378, 0.02075, 0.97880, 0.96106, 0.99995, 0.99992]
[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0] -> [0.00065, 0.00081, 0.00179, 0.00514, 0.27792, 0.27650, 0.66810, 0.25561, 0.88505, 0.88768, 0.99996, 0.99996]
 */