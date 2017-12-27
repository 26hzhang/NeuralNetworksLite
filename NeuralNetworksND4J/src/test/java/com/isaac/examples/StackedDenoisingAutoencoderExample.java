package com.isaac.examples;

import com.isaac.neuralnetworks.StackedDenoisingAutoencoder;
import com.isaac.utils.Evaluation;
import com.isaac.utils.RandomGenerator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

@SuppressWarnings("Duplicates")
public class StackedDenoisingAutoencoderExample {

	public static void main(String[] args) {
		final Random rng = new Random(1234);
		// Declare variables and constants
		int trainSetSizeEach = 200;
		int validateSetSizeEach = 200;
		int testSetSizeEach = 50;
		int nInEach = 20;
		double pNoiseTrain = 0.2;
		double pNoiseTest = 0.25;

		final int patterns = 3; // nOut

		final int trainSetSize = trainSetSizeEach * patterns;
		final int validateSetSize = validateSetSizeEach * patterns;
		final int testSetSize = testSetSizeEach * patterns;

		final int nIn = nInEach * patterns;
		int[] hiddenLayerSizes = {20, 20};
		double corruptionLevel = 0.3;

		INDArray trainSet = Nd4j.create(new double[trainSetSize * nIn], new int[] {trainSetSize, nIn});
		INDArray validateSet = Nd4j.create(new double[validateSetSize * nIn], new int[] {validateSetSize, nIn});
		INDArray validateLabel = Nd4j.create(new double[validateSetSize * patterns], new int[] {validateSetSize, patterns});
		INDArray testSet = Nd4j.create(new double[testSetSize * nIn], new int[] {testSetSize, nIn});
		INDArray testLabel = Nd4j.create(new double[testSetSize * patterns], new int[] {testSetSize, patterns});

		int pretrainEpochs = 1000;
		double pretrainLearningRate = 0.2;
		int finetuneEpochs = 1000;
		double finetuneLearningRate = 0.15;

		int minibatchSize = 50;
		final int trainSetMinibatchNumber = trainSetSize / minibatchSize;
		final int validateSetMinibatchNumber = validateSetSize / minibatchSize;

		List<INDArray> trainSetMinibatch = new ArrayList<>();
		List<INDArray> validateSetMinibatch = new ArrayList<>();
		List<INDArray> validateLabelMinibatch = new ArrayList<>();
		List<Integer> minibatchIndex = new ArrayList<>();
		for (int i = 0; i < trainSetSize; i++) minibatchIndex.add(i);
		Collections.shuffle(minibatchIndex, rng);

		// Create training data and test data for demo.
		for (int pattern = 0; pattern < patterns; pattern++) {
			for (int n = 0; n < trainSetSizeEach; n++) {
				int n_ = pattern * trainSetSizeEach + n;
				for (int i = 0; i < nIn; i++) {
					if ( (n_ >= trainSetSizeEach * pattern && n_ < trainSetSizeEach * (pattern + 1) ) &&
							(i >= nInEach * pattern && i < nInEach * (pattern + 1)) ) {
						trainSet.put(n_, i, Nd4j.scalar(RandomGenerator.binomial(1, 1 - pNoiseTrain, rng)
								* rng.nextDouble() * 0.5 + 0.5));
					} else {
						trainSet.put(n_, i, Nd4j.scalar(RandomGenerator.binomial(1, pNoiseTrain, rng)
								* rng.nextDouble() * 0.5 + 0.5));
					}
				}
			}

			for (int n = 0; n < validateSetSizeEach; n++) {
				int n_ = pattern * validateSetSizeEach + n;
				for (int i = 0; i < nIn; i++) {
					if ( (n_ >= validateSetSizeEach * pattern && n_ < validateSetSizeEach * (pattern + 1) ) &&
							(i >= nInEach * pattern && i < nInEach * (pattern + 1)) ) {
						validateSet.put(n_, i, Nd4j.scalar(RandomGenerator.binomial(1, 1 - pNoiseTrain, rng)
								* rng.nextDouble() * 0.5 + 0.5));
					} else {
						validateSet.put(n_, i, Nd4j.scalar(RandomGenerator.binomial(1, pNoiseTrain, rng)
								* rng.nextDouble() * .5 + .5));
					}
				}
				for (int i = 0; i < patterns; i++) {
					if (i == pattern) { validateLabel.put(n_, i, Nd4j.scalar(1)); }
					else { validateLabel.put(n_, i, Nd4j.scalar(0)); }
				}
			}

			for (int n = 0; n < testSetSizeEach; n++) {
				int n_ = pattern * testSetSizeEach + n;
				for (int i = 0; i < nIn; i++) {
					if ( (n_ >= testSetSizeEach * pattern && n_ < testSetSizeEach * (pattern + 1) ) &&
							(i >= nInEach * pattern && i < nInEach * (pattern + 1)) ) {
						testSet.put(n_, i, Nd4j.scalar(RandomGenerator.binomial(1, 1 - pNoiseTest, rng)
								* rng.nextDouble() * 0.5 + 0.5));
					} else {
						testSet.put(n_, i, Nd4j.scalar(RandomGenerator.binomial(1, pNoiseTest, rng)
								* rng.nextDouble() * 0.5 + 0.5));
					}
				}
				for (int i = 0; i < patterns; i++) {
					if (i == pattern) { testLabel.put(n_, i, Nd4j.scalar(1)); }
					else { testLabel.put(n_, i, Nd4j.scalar(0)); }
				}
			}
		}

		// create minibatches
		for (int i = 0; i < trainSetMinibatchNumber; i++) {
			INDArray tmp = Nd4j.create(new double[minibatchSize * nIn], new int[] {minibatchSize, nIn});
			for (int j = 0; j < minibatchSize; j++) {
				tmp.putRow(j, trainSet.getRow(minibatchIndex.get(i * minibatchSize + j)));
			}
			trainSetMinibatch.add(tmp);
		}
		for (int i = 0; i < validateSetMinibatchNumber; i++) {
			INDArray tmpX = Nd4j.create(new double[minibatchSize * nIn], new int[]{minibatchSize, nIn});
			INDArray tmpT = Nd4j.create(new double[minibatchSize * patterns], new int[]{minibatchSize, patterns});
			for (int j = 0; j < minibatchSize; j++) {
				tmpX.putRow(j, validateSet.getRow(minibatchIndex.get(i * minibatchSize + j)));
				tmpT.putRow(j, validateLabel.getRow(minibatchIndex.get(i * minibatchSize + j)));
			}
			validateSetMinibatch.add(tmpX);
			validateLabelMinibatch.add(tmpT);
		}

		// Build Stacked Denoising Autoencoder model
		System.out.print("Building the model...");
		StackedDenoisingAutoencoder classifier = new StackedDenoisingAutoencoder(nIn, hiddenLayerSizes, patterns, rng);
		System.out.println("done.");

		// pre-training the model
		System.out.print("Pre-training the model...");
		classifier.preTrain(trainSetMinibatch, minibatchSize, trainSetMinibatchNumber, pretrainEpochs, pretrainLearningRate,
				corruptionLevel);
		System.out.println("done.");

		// fine-tuning the model
		System.out.print("Fine-tuning the model...");
		for (int epoch = 0; epoch < finetuneEpochs; epoch++) {
			for (int batch = 0; batch < validateSetMinibatchNumber; batch++) {
				classifier.finetune(validateSetMinibatch.get(batch), validateLabelMinibatch.get(batch), minibatchSize,
						finetuneLearningRate);
			}
			finetuneLearningRate *= 0.98;
		}
		System.out.println("done.");

		// test
		INDArray predicted_T = classifier.predict(testSet);

		// Evaluate the model
		Evaluation evaluation = new Evaluation(predicted_T, testLabel).fit();
		double accuracy = evaluation.getAccuracy();
		double[] precision = evaluation.getPrecision();
		double[] recall = evaluation.getRecall();

		System.out.println("SDA model evaluation");
		System.out.println("--------------------");
		System.out.printf("Accuracy: %.1f %%\n", accuracy * 100);
		System.out.println("Precision:");
		for (int i = 0; i < patterns; i++) {
			System.out.printf(" class %d: %.1f %%\n", i+1, precision[i] * 100);
		}
		System.out.println("Recall:");
		for (int i = 0; i < patterns; i++) {
			System.out.printf(" class %d: %.1f %%\n", i+1, recall[i] * 100);
		}
	}
}
/*
StackedDenoisingAutoencoderExampleDL4J model evaluation
-------------------------------------------------------
Accuracy: 98.7 %
Precision:
 class 1: 98.0 %
 class 2: 100.0 %
 class 3: 98.0 %
Recall:
 class 1: 100.0 %
 class 2: 96.0 %
 class 3: 100.0 %
 */