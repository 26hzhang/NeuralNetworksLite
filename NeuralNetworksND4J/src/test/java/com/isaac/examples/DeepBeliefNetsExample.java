package com.isaac.examples;

import com.isaac.neuralnetworks.DeepBeliefNets;
import com.isaac.utils.Evaluation;
import com.isaac.utils.RandomGenerator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

@SuppressWarnings("Duplicates")
public class DeepBeliefNetsExample {

	public static void main(String[] args) {
		final Random rng = new Random(1234);
        // Declare variables and constants
		int trainSetSizeEach = 200;        // for demo
		int validateSetSizeEach = 200;   // for demo
		int testSetSizeEach = 50;          // for demo
		int nInEach = 20;             // for demo
		double pNoiseTrain = 0.2;  // for demo
		double pNoiseTest = 0.25;     // for demo

		final int patterns = 3; // nOut
		final int trainSetSize = trainSetSizeEach * patterns;
		final int validateSetSize = validateSetSizeEach * patterns;
		final int testSetSize = testSetSizeEach * patterns;
		final int nIn = nInEach * patterns;
		int[] hiddenLayerSizes = {20, 20};
		final int k = 1;  // CD-k in RBM

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
						trainSet.put(n_, i, Nd4j.scalar(RandomGenerator.binomial(1, 1 - pNoiseTrain, rng)));
					} else {
						trainSet.put(n_, i, Nd4j.scalar(RandomGenerator.binomial(1, pNoiseTrain, rng)));
					}
				}
			}
			for (int n = 0; n < validateSetSizeEach; n++) {
				int n_ = pattern * validateSetSizeEach + n;
				for (int i = 0; i < nIn; i++) {
					if ( (n_ >= validateSetSizeEach * pattern && n_ < validateSetSizeEach * (pattern + 1) ) &&
							(i >= nInEach * pattern && i < nInEach * (pattern + 1)) ) {
						validateSet.put(n_, i,
								Nd4j.scalar((double) RandomGenerator.binomial(1, 1 - pNoiseTrain, rng)));
					} else {
						validateSet.put(n_, i,
								Nd4j.scalar((double) RandomGenerator.binomial(1, pNoiseTrain, rng)));
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
						testSet.put(n_, i,
								Nd4j.scalar((double) RandomGenerator.binomial(1, 1 - pNoiseTest, rng)));
					} else {
						testSet.put(n_, i,
								Nd4j.scalar((double) RandomGenerator.binomial(1, pNoiseTest, rng)));
					}
				}
				for (int i = 0; i < patterns; i++) {
					if (i == pattern) { testLabel.put(n_, i,  Nd4j.scalar(1)); }
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

		// Build Deep Belief Nets model
		System.out.print("Building the model...");
		DeepBeliefNets classifier = new DeepBeliefNets(nIn, hiddenLayerSizes, patterns, rng);
		System.out.println("done.");

		// pre-training the model
		System.out.print("Pre-training the model...");
		classifier.pretrain(trainSetMinibatch, minibatchSize, trainSetMinibatchNumber, pretrainEpochs, pretrainLearningRate, k);
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
		INDArray predictLabel = classifier.predict(testSet);

		// Evaluate the model
		Evaluation evaluation = new Evaluation(predictLabel, testLabel).fit();
		double accuracy = evaluation.getAccuracy();
		double[] precision = evaluation.getPrecision();
		double[] recall = evaluation.getRecall();

		System.out.println("DBN model evaluation");
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