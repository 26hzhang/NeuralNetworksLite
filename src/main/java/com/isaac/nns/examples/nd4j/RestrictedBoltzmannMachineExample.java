package com.isaac.nns.examples.nd4j;

import com.isaac.nns.initialization.Activation;
import com.isaac.nns.initialization.Weight;
import com.isaac.nns.layers.nd4j.RestrictedBoltzmannMachine;
import com.isaac.nns.utils.RandomGenerator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

public class RestrictedBoltzmannMachineExample {
	public static void main(String[] args) {
		final Random rng = new Random(123); // seed random
		/*
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

		INDArray train_X = Nd4j.create(new double[train_N * nVisible], new int[] {train_N, nVisible});
		INDArray test_X = Nd4j.create(new double[test_N * nVisible], new int[] {test_N, nVisible});
		INDArray reconstructed_X = Nd4j.create(new double[test_N * nVisible], new int[] {test_N, nVisible});

		int epochs = 1000;
		double learningRate = 0.2;
		int minibatchSize = 10;
		final int minibatch_N = train_N / minibatchSize;

		List<INDArray> train_X_minibatch = new ArrayList<>();
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
						train_X.put(n_, i, RandomGenerator.binomial(1, 1-pNoise_Training, rng));
					} else {
						train_X.put(n_, i, RandomGenerator.binomial(1, pNoise_Training, rng));
					}
				}
			}
			for (int n = 0; n < test_N_each; n++) {
				int n_ = pattern * test_N_each + n;
				for (int i = 0; i < nVisible; i++) {
					if ((n_ >= test_N_each * pattern && n_ < test_N_each * (pattern + 1)) &&
							(i >= nVisible_each * pattern && i < nVisible_each * (pattern + 1))) {
						test_X.put(n_, i, RandomGenerator.binomial(1, 1-pNoise_Test, rng));
					} else {
						test_X.put(n_, i, RandomGenerator.binomial(1, pNoise_Test, rng));
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

		/*
         * Build Restricted Boltzmann Machines model
         */
		RestrictedBoltzmannMachine nn = new RestrictedBoltzmannMachine(nVisible, nHidden, Weight.UNIFORM, Activation.Sigmoid,null);

		// train with contrastive divergence
		for (int epoch = 0; epoch < epochs; epoch++) {
			for (int batch = 0; batch < minibatch_N; batch++) {
				nn.contrastiveDivergence(train_X_minibatch.get(batch), minibatchSize, learningRate, k);
			}
			learningRate *= 0.995; // update learningRate
		}

		// test (reconstruct noised data)
		reconstructed_X = nn.reconstruct(test_X);

		// evaluation
		System.out.println("RBM model reconstruction evaluation");
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
RBM model reconstruction evaluation
-----------------------------------
Class1
[1.00, 1.00, 1.00, 1.00, 1.00, 0.00, 0.00, 1.00, 1.00, 0.00, 0.00, 0.00] -> [0.97391, 0.93102, 0.93623, 0.96150, 0.07656, 0.07372, 0.04244, 0.03858, 0.03732, 0.04396, 0.05795, 0.04642]
[1.00, 1.00, 0.00, 1.00, 1.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00] -> [0.94435, 0.88525, 0.84971, 0.93479, 0.16380, 0.14450, 0.09752, 0.07689, 0.02579, 0.03053, 0.03613, 0.02930]

Class2
[1.00, 0.00, 1.00, 0.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.00, 0.00, 1.00] -> [0.06715, 0.07540, 0.09655, 0.07426, 0.89135, 0.87820, 0.85225, 0.86972, 0.09200, 0.11258, 0.09776, 0.05849]
[0.00, 0.00, 0.00, 1.00, 0.00, 1.00, 1.00, 1.00, 1.00, 0.00, 1.00, 0.00] -> [0.07325, 0.08934, 0.10592, 0.09044, 0.83042, 0.82436, 0.79051, 0.80984, 0.14219, 0.16801, 0.14321, 0.09179]

Class3
[0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 1.00, 1.00, 1.00, 0.00, 0.00] -> [0.10097, 0.14175, 0.18835, 0.08949, 0.08835, 0.14112, 0.11679, 0.12215, 0.76314, 0.81385, 0.78852, 0.71471]
[0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 1.00, 0.00, 1.00, 0.00] -> [0.02820, 0.05305, 0.05864, 0.02012, 0.05975, 0.10760, 0.10705, 0.09209, 0.86379, 0.90844, 0.88168, 0.83640]
 */