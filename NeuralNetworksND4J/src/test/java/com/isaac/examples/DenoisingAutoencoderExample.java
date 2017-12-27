package com.isaac.examples;

import com.isaac.initialization.Activation;
import com.isaac.layers.DenoisingAutoencoder;
import com.isaac.utils.RandomGenerator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
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
		int nVisible_each = 4;
		double pNoise_Training = 0.05;
		double pNoise_Test = 0.25;

		final int patterns = 3;
		final int trainSetSize = trainSetSizeEach * patterns;
		final int testSetSize = testSetSizeEach * patterns;
		final int nVisible = nVisible_each * patterns;
		int nHidden = 6;
		double corruptionLevel = 0.3;

		INDArray trainSet = Nd4j.create(new double[trainSetSize * nVisible], new int[] {trainSetSize, nVisible});
		INDArray testSet = Nd4j.create(new double[testSetSize * nVisible], new int[] {testSetSize, nVisible});

		int epochs = 1000;
		double learningRate = 0.2;
		int minibatchSize = 10;
		final int minibatch_N = trainSetSize / minibatchSize;

		List<INDArray> trainSetMinibatch = new ArrayList<>();
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
							(i >= nVisible_each * pattern && i < nVisible_each * (pattern + 1)) ) {
						trainSet.put(n_, i, Nd4j.scalar(RandomGenerator.binomial(1, 1 - pNoise_Training, rng)));
					} else {
						trainSet.put(n_, i, Nd4j.scalar(RandomGenerator.binomial(1, pNoise_Training, rng)));
					}
				}
			}
			for (int n = 0; n < testSetSizeEach; n++) {
				int n_ = pattern * testSetSizeEach + n;
				for (int i = 0; i < nVisible; i++) {
					if ( (n_ >= testSetSizeEach * pattern && n_ < testSetSizeEach * (pattern + 1) ) &&
							(i >= nVisible_each * pattern && i < nVisible_each * (pattern + 1)) ) {
						testSet.put(n_, i, Nd4j.scalar(RandomGenerator.binomial(1, 1 - pNoise_Test, rng)));
					} else {
						testSet.put(n_, i, Nd4j.scalar(RandomGenerator.binomial(1, pNoise_Test, rng)));
					}
				}
			}
		}

		// create minibatches
		for (int i = 0; i < minibatch_N; i++) {
			INDArray tmp = Nd4j.create(new double[minibatchSize * nVisible], new int[] {minibatchSize, nVisible});
			for (int j = 0; j < minibatchSize; j++) {
				tmp.putRow(j, trainSet.getRow(minibatchIndex.get(i * minibatchSize + j)));
			}
			trainSetMinibatch.add(tmp);
		}

		// Build Denoising Autoencoder Model
		// construct DA
		DenoisingAutoencoder nn = new DenoisingAutoencoder(nVisible, nHidden, null, null, null, null,
				Activation.Sigmoid);

		// train
		for (int epoch = 0; epoch < epochs; epoch++) {
			for (int batch = 0; batch < minibatch_N; batch++) {
				nn.train(trainSetMinibatch.get(batch), minibatchSize, learningRate, corruptionLevel);
			}
		}

		// test
		INDArray reconstructed_X = nn.reconstruct(testSet);

		// evaluation
		System.out.println("DA model reconstruction evaluation");
		System.out.println("-----------------------------------");
		for (int pattern = 0; pattern < patterns; pattern++) {
			System.out.printf("Class%d\n", pattern + 1);
			for (int n = 0; n < testSetSizeEach; n++) {
				int n_ = pattern * testSetSizeEach + n;
				System.out.print(testSet.getRow(n_) + " -> ");
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