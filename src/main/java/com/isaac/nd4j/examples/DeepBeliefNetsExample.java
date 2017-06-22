package com.isaac.nd4j.examples;

import com.isaac.nd4j.neuralnetworks.DeepBeliefNets;
import com.isaac.nd4j.utils.RandomGenerator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

public class DeepBeliefNetsExample {

	public static void main(String[] args) {
		final Random rng = new Random(123);
        /*
         * Declare variables and constants
         */
		int train_N_each = 200;        // for demo
		int validation_N_each = 200;   // for demo
		int test_N_each = 50;          // for demo
		int nIn_each = 20;             // for demo
		double pNoise_Training = 0.2;  // for demo
		double pNoise_Test = 0.25;     // for demo

		final int patterns = 3;
		final int train_N = train_N_each * patterns;
		final int validation_N = validation_N_each * patterns;
		final int test_N = test_N_each * patterns;
		final int nIn = nIn_each * patterns;
		final int nOut = patterns;
		int[] hiddenLayerSizes = {20, 20};
		final int k = 1;  // CD-k in RBM

		INDArray train_X = Nd4j.create(new double[train_N * nIn], new int[] {train_N, nIn});
		INDArray validation_X = Nd4j.create(new double[validation_N * nIn], new int[] {validation_N, nIn});
		INDArray validation_T = Nd4j.create(new double[validation_N * nOut], new int[] {validation_N, nOut});
		INDArray test_X = Nd4j.create(new double[test_N * nIn], new int[] {test_N, nIn});
		INDArray test_T = Nd4j.create(new double[test_N * nOut], new int[] {test_N, nOut});

		int pretrainEpochs = 1000;
		double pretrainLearningRate = 0.2;
		int finetuneEpochs = 1000;
		double finetuneLearningRate = 0.15;
		int minibatchSize = 50;
		final int train_minibatch_N = train_N / minibatchSize;
		final int validation_minibatch_N = validation_N / minibatchSize;

		List<INDArray> train_X_minibatch = new ArrayList<>();
		List<INDArray> validation_X_minibatch = new ArrayList<>();
		List<INDArray> validation_T_minibatch = new ArrayList<>();
		List<Integer> minibatchIndex = new ArrayList<>();
		for (int i = 0; i < train_N; i++) minibatchIndex.add(i);
		Collections.shuffle(minibatchIndex, rng);

		/*
         * Create training data and test data for demo.
         */
		for (int pattern = 0; pattern < patterns; pattern++) {
			for (int n = 0; n < train_N_each; n++) {
				int n_ = pattern * train_N_each + n;
				for (int i = 0; i < nIn; i++) {
					if ( (n_ >= train_N_each * pattern && n_ < train_N_each * (pattern + 1) ) &&
							(i >= nIn_each * pattern && i < nIn_each * (pattern + 1)) ) {
						train_X.put(n_, i, Nd4j.scalar(RandomGenerator.binomial(1, 1 - pNoise_Training, rng)));
					} else {
						train_X.put(n_, i, Nd4j.scalar(RandomGenerator.binomial(1, pNoise_Training, rng)));
					}
				}
			}
			for (int n = 0; n < validation_N_each; n++) {
				int n_ = pattern * validation_N_each + n;
				for (int i = 0; i < nIn; i++) {
					if ( (n_ >= validation_N_each * pattern && n_ < validation_N_each * (pattern + 1) ) &&
							(i >= nIn_each * pattern && i < nIn_each * (pattern + 1)) ) {
						validation_X.put(n_, i,
								Nd4j.scalar((double) RandomGenerator.binomial(1, 1 - pNoise_Training, rng)));
					} else {
						validation_X.put(n_, i,
								Nd4j.scalar((double) RandomGenerator.binomial(1, pNoise_Training, rng)));
					}
				}
				for (int i = 0; i < nOut; i++) {
					if (i == pattern) {
						validation_T.put(n_, i, Nd4j.scalar(1));
					} else {
						validation_T.put(n_, i, Nd4j.scalar(0));
					}
				}
			}
			for (int n = 0; n < test_N_each; n++) {
				int n_ = pattern * test_N_each + n;
				for (int i = 0; i < nIn; i++) {
					if ( (n_ >= test_N_each * pattern && n_ < test_N_each * (pattern + 1) ) &&
							(i >= nIn_each * pattern && i < nIn_each * (pattern + 1)) ) {
						test_X.put(n_, i,
								Nd4j.scalar((double) RandomGenerator.binomial(1, 1 - pNoise_Test, rng)));
					} else {
						test_X.put(n_, i,
								Nd4j.scalar((double) RandomGenerator.binomial(1, pNoise_Test, rng)));
					}
				}
				for (int i = 0; i < nOut; i++) {
					if (i == pattern) {
						test_T.put(n_, i,  Nd4j.scalar(1));
					} else {
						test_T.put(n_, i, Nd4j.scalar(0));
					}
				}
			}
		}

		// create minibatches
		for (int i = 0; i < train_minibatch_N; i++) {
			INDArray tmp = Nd4j.create(new double[minibatchSize * nIn], new int[] {minibatchSize, nIn});
			for (int j = 0; j < minibatchSize; j++) {
				tmp.putRow(j, train_X.getRow(minibatchIndex.get(i * minibatchSize + j)));
			}
			train_X_minibatch.add(tmp);
		}
		for (int i = 0; i < validation_minibatch_N; i++) {
			INDArray tmpX = Nd4j.create(new double[minibatchSize * nIn], new int[]{minibatchSize, nIn});
			INDArray tmpT = Nd4j.create(new double[minibatchSize * nOut], new int[]{minibatchSize, nOut});
			for (int j = 0; j < minibatchSize; j++) {
				tmpX.putRow(j, validation_X.getRow(minibatchIndex.get(i * minibatchSize + j)));
				tmpT.putRow(j, validation_T.getRow(minibatchIndex.get(i * minibatchSize + j)));
			}
			validation_X_minibatch.add(tmpX);
			validation_T_minibatch.add(tmpT);
		}

		/*
         * Build Deep Belief Nets model
         */
		// construct DBN
		System.out.print("Building the model...");
		DeepBeliefNets classifier = new DeepBeliefNets(nIn, hiddenLayerSizes, nOut, rng);
		System.out.println("done.");

		// pre-training the model
		System.out.print("Pre-training the model...");
		classifier.pretrain(train_X_minibatch, minibatchSize, train_minibatch_N, pretrainEpochs, pretrainLearningRate, k);
		System.out.println("done.");

		// fine-tuning the model
		System.out.print("Fine-tuning the model...");
		for (int epoch = 0; epoch < finetuneEpochs; epoch++) {
			for (int batch = 0; batch < validation_minibatch_N; batch++) {
				classifier.finetune(validation_X_minibatch.get(batch), validation_T_minibatch.get(batch), minibatchSize,
						finetuneLearningRate);
			}
			finetuneLearningRate *= 0.98;
		}
		System.out.println("done.");

		// test
		INDArray predicted_T = classifier.predict(test_X);

		/*
         * Evaluate the model
         */
		int[][] confusionMatrix = new int[patterns][patterns];
		double accuracy = 0.;
		double[] precision = new double[patterns];
		double[] recall = new double[patterns];
		for (int i = 0; i < test_N; i++) {
			int predicted_ = 0;
			int actual_ = 0;
			for (int j = 0; j < nOut; j++) {
				if (predicted_T.getDouble(i, j) == 1.0)
					predicted_ = j;
			}
			for (int j = 0; j < nOut; j++) {
				if (test_T.getDouble(i, j) == 1.0)
					actual_ = j;
			}
			confusionMatrix[actual_][predicted_] += 1;
		}

		for (int i = 0; i < patterns; i++) {
			double col_ = 0.;
			double row_ = 0.;
			for (int j = 0; j < patterns; j++) {
				if (i == j) {
					accuracy += confusionMatrix[i][j];
					precision[i] += confusionMatrix[j][i];
					recall[i] += confusionMatrix[i][j];
				}
				col_ += confusionMatrix[j][i];
				row_ += confusionMatrix[i][j];
			}
			precision[i] /= col_;
			recall[i] /= row_;
		}
		accuracy /= test_N;

		System.out.println("--------------------");
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
/*
DBN model evaluation
--------------------
Accuracy: 100.0 %
Precision:
 class 1: 100.0 %
 class 2: 100.0 %
 class 3: 100.0 %
Recall:
 class 1: 100.0 %
 class 2: 100.0 %
 class 3: 100.0 %
 */