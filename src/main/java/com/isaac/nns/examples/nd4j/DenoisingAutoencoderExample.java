package com.isaac.nns.examples.nd4j;

import com.isaac.nns.initialization.Activation;
import com.isaac.nns.initialization.Weight;
import com.isaac.nns.layers.nd4j.DenoisingAutoencoder;
import com.isaac.nns.utils.RandomGenerator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
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

		INDArray train_X = Nd4j.create(new double[train_N * nVisible], new int[] {train_N, nVisible});
		INDArray test_X = Nd4j.create(new double[test_N * nVisible], new int[] {test_N, nVisible});

		int epochs = 1000;
		double learningRate = 0.2;
		int minibatchSize = 10;
		final int minibatch_N = train_N / minibatchSize;

		List<INDArray> train_X_minibatch = new ArrayList<>();
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
						train_X.put(n_, i, Nd4j.scalar(RandomGenerator.binomial(1, 1 - pNoise_Training, rng)));
					} else {
						train_X.put(n_, i, Nd4j.scalar(RandomGenerator.binomial(1, pNoise_Training, rng)));
					}
				}
			}
			for (int n = 0; n < test_N_each; n++) {
				int n_ = pattern * test_N_each + n;
				for (int i = 0; i < nVisible; i++) {
					if ( (n_ >= test_N_each * pattern && n_ < test_N_each * (pattern + 1) ) &&
							(i >= nVisible_each * pattern && i < nVisible_each * (pattern + 1)) ) {
						test_X.put(n_, i, Nd4j.scalar(RandomGenerator.binomial(1, 1 - pNoise_Test, rng)));
					} else {
						test_X.put(n_, i, Nd4j.scalar(RandomGenerator.binomial(1, pNoise_Test, rng)));
					}
				}
			}
		}

		// create minibatches
		for (int i = 0; i < minibatch_N; i++) {
			INDArray tmp = Nd4j.create(new double[minibatchSize * nVisible], new int[] {minibatchSize, nVisible});
			for (int j = 0; j < minibatchSize; j++) {
				tmp.putRow(j, train_X.getRow(minibatchIndex.get(i * minibatchSize + j)));
			}
			train_X_minibatch.add(tmp);
		}

		// Build Denoising Autoencoder Model
		// construct DA
		DenoisingAutoencoder nn = new DenoisingAutoencoder(nVisible, nHidden, Weight.UNIFORM, Activation.Sigmoid, null);

		// train
		for (int epoch = 0; epoch < epochs; epoch++) {
			for (int batch = 0; batch < minibatch_N; batch++) {
				nn.train(train_X_minibatch.get(batch), minibatchSize, learningRate, corruptionLevel);
			}
		}

		// test
		INDArray reconstructed_X = nn.reconstruct(test_X);

		// evaluation
		System.out.println("-----------------------------------");
		System.out.println("DA model reconstruction evaluation");
		System.out.println("-----------------------------------");
		for (int pattern = 0; pattern < patterns; pattern++) {
			System.out.printf("Class%d\n", pattern + 1);
			for (int n = 0; n < test_N_each; n++) {
				int n_ = pattern * test_N_each + n;
				System.out.print(test_X.getRow(n_) + " -> ");
				System.out.print("[");
				for (int i = 0; i < nVisible-1; i++) {
					System.out.printf("%.5f, ", reconstructed_X.getDouble(n_, i));
				}
				System.out.printf("%.5f]\n", reconstructed_X.getDouble(n_, nVisible - 1));
			}
			System.out.println();
		}
	}
}
/*
DA model reconstruction evaluation
-----------------------------------
Class1
[1.00, 1.00, 1.00, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00] -> [0.99143, 0.96950, 0.94878, 0.96128, 0.25164, 0.00689, 0.28337, 0.01260, 0.00628, 0.02127, 0.02903, 0.00754]
[1.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00] -> [0.98497, 0.58425, 0.77636, 0.64380, 0.54206, 0.13014, 0.25308, 0.90099, 0.00345, 0.00763, 0.00192, 0.01826]

Class2
[1.00, 1.00, 0.00, 0.00, 1.00, 1.00, 0.00, 1.00, 0.00, 0.00, 0.00, 1.00] -> [0.77808, 0.27961, 0.50884, 0.56585, 0.94894, 0.77863, 0.87677, 0.88417, 0.00127, 0.00635, 0.00380, 0.01507]
[0.00, 0.00, 0.00, 1.00, 0.00, 1.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00] -> [0.02933, 0.35982, 0.57685, 0.60923, 0.79825, 0.99688, 0.61614, 0.90188, 0.00206, 0.00678, 0.00234, 0.01556]

Class3
[0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 1.00, 1.00, 1.00] -> [0.06774, 0.03027, 0.03509, 0.04168, 0.00268, 0.01335, 0.01081, 0.01293, 0.98481, 0.97507, 0.93744, 0.95753]
[0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 1.00, 0.00, 1.00, 1.00, 1.00, 1.00] -> [0.00050, 0.00254, 0.00468, 0.00825, 0.40695, 0.81957, 0.79200, 0.08514, 0.90997, 0.84760, 0.94842, 0.84017]
 */