package com.isaac.examples;

import com.isaac.neuralnetworks.StackedDenoisingAutoencoder;
import com.isaac.utils.Evaluation;
import com.isaac.utils.RandomGenerator;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

@SuppressWarnings("Duplicates")
public class StackedDenoisingAutoencoderExample {

	public static void main(String[] args) {
		final Random rng = new Random(123);
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

        double[][] trainSet = new double[trainSetSize][nIn];

        double[][] validateSet = new double[validateSetSize][nIn];
        int[][] validateLabel = new int[validateSetSize][patterns];

        double[][] testSet = new double[testSetSize][nIn];
        Integer[][] testLabel = new Integer[testSetSize][patterns];
        Integer[][] predictLabel = new Integer[testSetSize][patterns];

        int pretrainEpochs = 1000;
        double pretrainLearningRate = 0.2;
        int finetuneEpochs = 1000;
        double finetuneLearningRate = 0.15;

        int minibatchSize = 50;
        final int trainSetMinibatchNumber = trainSetSize / minibatchSize;
        final int validateSetMinibatchNumber = validateSetSize / minibatchSize;

        double[][][] trainSetMinibatch = new double[trainSetMinibatchNumber][minibatchSize][nIn];
        double[][][] validateSetMinibatch = new double[validateSetMinibatchNumber][minibatchSize][nIn];
        int[][][] validateLabelMinibatch = new int[validateSetMinibatchNumber][minibatchSize][patterns];
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
                        trainSet[n_][i] = RandomGenerator.binomial(1, 1 - pNoiseTrain, rng) * rng.nextDouble() * .5 + .5;
                    } else { trainSet[n_][i] = RandomGenerator.binomial(1, pNoiseTrain, rng) * rng.nextDouble() * .5 + .5; }
                }
            }
            for (int n = 0; n < validateSetSizeEach; n++) {
                int n_ = pattern * validateSetSizeEach + n;
                for (int i = 0; i < nIn; i++) {
                    if ( (n_ >= validateSetSizeEach * pattern && n_ < validateSetSizeEach * (pattern + 1) ) &&
                            (i >= nInEach * pattern && i < nInEach * (pattern + 1)) ) {
                        validateSet[n_][i] = (double) RandomGenerator.binomial(1, 1 - pNoiseTrain, rng) *
                                rng.nextDouble() * .5 + .5;
                    } else {
                        validateSet[n_][i] = (double) RandomGenerator.binomial(1, pNoiseTrain, rng) * rng.nextDouble() * .5 + .5;
                    }
                }
                for (int i = 0; i < patterns; i++) {
                    if (i == pattern) { validateLabel[n_][i] = 1;}
                    else { validateLabel[n_][i] = 0; }
                }
            }
            for (int n = 0; n < testSetSizeEach; n++) {
                int n_ = pattern * testSetSizeEach + n;
                for (int i = 0; i < nIn; i++) {
                    if ( (n_ >= testSetSizeEach * pattern && n_ < testSetSizeEach * (pattern + 1) ) &&
                            (i >= nInEach * pattern && i < nInEach * (pattern + 1)) ) {
                        testSet[n_][i] = (double) RandomGenerator.binomial(1, 1 - pNoiseTest, rng) *
                                rng.nextDouble() * .5 + .5;
                    } else {
                        testSet[n_][i] = (double) RandomGenerator.binomial(1, pNoiseTest, rng) * rng.nextDouble() * .5 + .5;
                    }
                }
                for (int i = 0; i < patterns; i++) {
                    if (i == pattern) { testLabel[n_][i] = 1; }
                    else { testLabel[n_][i] = 0; }
                }
            }
        }

        // create minibatches
        for (int j = 0; j < minibatchSize; j++) {
            for (int i = 0; i < trainSetMinibatchNumber; i++) {
                trainSetMinibatch[i][j] = trainSet[minibatchIndex.get(i * minibatchSize + j)];
            }
            for (int i = 0; i < validateSetMinibatchNumber; i++) {
                validateSetMinibatch[i][j] = validateSet[minibatchIndex.get(i * minibatchSize + j)];
                validateLabelMinibatch[i][j] = validateLabel[minibatchIndex.get(i * minibatchSize + j)];
            }
        }

        // Build Stacked Denoising AutoEncoder model
        System.out.print("Building the model...");
        StackedDenoisingAutoencoder classifier = new StackedDenoisingAutoencoder(nIn, hiddenLayerSizes, patterns, rng);
        System.out.println(" done.");


        // pre-training the model
        System.out.print("Pre-training the model...");
        classifier.pretrain(trainSetMinibatch, minibatchSize, trainSetMinibatchNumber, pretrainEpochs, pretrainLearningRate,
                corruptionLevel);
        System.out.println("done.");


        // fine-tuning the model
        System.out.print("Fine-tuning the model...");
        for (int epoch = 0; epoch < finetuneEpochs; epoch++) {
            for (int batch = 0; batch < validateSetMinibatchNumber; batch++) {
                classifier.finetune(validateSetMinibatch[batch], validateLabelMinibatch[batch], minibatchSize,
                        finetuneLearningRate);
            }
            finetuneLearningRate *= 0.98;
        }
        System.out.println("done.");


        // test
        for (int i = 0; i < testSetSize; i++) predictLabel[i] = classifier.predict(testSet[i]);

        // Evaluate the model
        Evaluation evaluation = new Evaluation(predictLabel, testLabel).fit();
        double accuracy = evaluation.getAccuracy();
        double[] precision = evaluation.getPrecision();
        double[] recall = evaluation.getRecall();

        System.out.println("SDA model evaluation");
        System.out.println("--------------------");
        System.out.printf("Accuracy: %.1f %%\n", accuracy * 100);
        System.out.println("Precision:");
        for (int i = 0; i < patterns; i++) System.out.printf(" class %d: %.1f %%\n", i+1, precision[i] * 100);
        System.out.println("Recall:");
        for (int i = 0; i < patterns; i++) System.out.printf(" class %d: %.1f %%\n", i+1, recall[i] * 100);
	}

}