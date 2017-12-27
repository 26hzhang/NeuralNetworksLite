package com.isaac.examples;

import com.isaac.initialization.Activation;
import com.isaac.layers.RestrictedBoltzmannMachine;
import com.isaac.utils.RandomGenerator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

@SuppressWarnings("Duplicates")
public class RestrictedBoltzmannMachineExample {
	public static void main(String[] args) {
		final Random rng = new Random(123); // seed random
		// declare variables and constants
		int trainSetSizeEach = 200;
		int testSetSizeEach = 2;
		int nVisibleEach = 4;
		double pNoiseTrain = 0.05; // train data noise
		double pNoiseTest = 0.25; // test data noise
		int k = 1; // k times sampling of contrastive divergence

		final int patterns = 3;
		final int trainSetSize = trainSetSizeEach * patterns;
		final int testSetSize = testSetSizeEach * patterns;
		final int nVisible = nVisibleEach * patterns;
		int nHidden = 6;

		INDArray trainSet = Nd4j.create(new double[trainSetSize * nVisible], new int[] {trainSetSize, nVisible});
		INDArray testSet = Nd4j.create(new double[testSetSize * nVisible], new int[] {testSetSize, nVisible});
		INDArray reconstructedSet;

		int epochs = 1000;
		double learningRate = 0.2;
		int minibatchSize = 10;
		final int minibatchNumber = trainSetSize / minibatchSize;

		List<INDArray> trainSetMinibatch = new ArrayList<>();
		List<Integer> minibatchIndex = new ArrayList<>();
		for (int i = 0; i < trainSetSize; i++) minibatchIndex.add(i);
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
			for (int n = 0; n < trainSetSizeEach; n++) {
				int n_ = pattern * trainSetSizeEach + n;
				for (int i = 0; i < nVisible; i++) {
					if ( (n_ >= trainSetSizeEach * pattern && n_ < trainSetSizeEach * (pattern + 1) ) &&
							(i >= nVisibleEach * pattern && i < nVisibleEach * (pattern + 1)) ) {
						trainSet.put(n_, i, RandomGenerator.binomial(1, 1-pNoiseTrain, rng));
					} else {
						trainSet.put(n_, i, RandomGenerator.binomial(1, pNoiseTrain, rng));
					}
				}
			}
			for (int n = 0; n < testSetSizeEach; n++) {
				int n_ = pattern * testSetSizeEach + n;
				for (int i = 0; i < nVisible; i++) {
					if ((n_ >= testSetSizeEach * pattern && n_ < testSetSizeEach * (pattern + 1)) &&
							(i >= nVisibleEach * pattern && i < nVisibleEach * (pattern + 1))) {
						testSet.put(n_, i, RandomGenerator.binomial(1, 1-pNoiseTest, rng));
					} else {
						testSet.put(n_, i, RandomGenerator.binomial(1, pNoiseTest, rng));
					}
				}
			}
		}

		// create minibatches
		for (int i = 0; i < minibatchNumber; i++) {
			INDArray tmp = Nd4j.create(new double[minibatchSize * nVisible], new int[] {minibatchSize, nVisible});
			for (int j = 0; j < minibatchSize; j++) {
				tmp.putRow(j, trainSet.getRow(minibatchIndex.get(i * minibatchSize + j)));
			}
			trainSetMinibatch.add(tmp);
		}

		// Build Restricted Boltzmann Machines model
		RestrictedBoltzmannMachine nn = new RestrictedBoltzmannMachine(nVisible, nHidden, null, null, null,
				rng, Activation.Sigmoid);

		// train with contrastive divergence
		for (int epoch = 0; epoch < epochs; epoch++) {
			for (int batch = 0; batch < minibatchNumber; batch++) {
				nn.contrastiveDivergence(trainSetMinibatch.get(batch), minibatchSize, learningRate, k);
			}
			learningRate *= 0.995; // update learningRate
		}

		// test (reconstruct noised data)
		reconstructedSet = nn.reconstruct(testSet);

		// evaluation
		System.out.println("RBM model reconstruction evaluation");
		for (int pattern = 0; pattern < patterns; pattern++) {
			System.out.printf("Class%d\n", pattern + 1);
			for (int n = 0; n < testSetSizeEach; n++) {
				int n_ = pattern * testSetSizeEach + n;
				System.out.print(testSet.getRow(n_) + " -> ");
				System.out.print("[");
				for (int i = 0; i < nVisible-1; i++) System.out.printf("%.5f, ", reconstructedSet.getDouble(n_, i));
				System.out.printf("%.5f]\n", reconstructedSet.getDouble(n_, nVisible - 1));
			}
			System.out.println();
		}
	}
}