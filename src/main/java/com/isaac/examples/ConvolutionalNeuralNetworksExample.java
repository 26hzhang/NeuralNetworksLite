package com.isaac.examples;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import com.isaac.neuralnetworks.ConvolutionalNeuralNetworks;
import com.isaac.utils.RandomGenerator;

public class ConvolutionalNeuralNetworksExample {

	public static void main(String[] args) {
		final Random rng = new Random(123);  // seed random
        //
        // Declare variables and constants
        //
        int train_N_each = 50;        // for demo
        int test_N_each = 10;          // for demo
        double pNoise_Training = 0.05;  // for demo
        double pNoise_Test = 0.10;     // for demo
        final int patterns = 3;
        final int train_N = train_N_each * patterns;
        final int test_N = test_N_each * patterns;
        final int[] imageSize = {12, 12};
        final int channel = 1;
        int[] nKernels = {10, 20};
        int[][] kernelSizes = { {3, 3}, {2, 2} };
        int[][] poolSizes = { {2, 2}, {2, 2} };

        int nHidden = 20;
        final int nOut = patterns;

        double[][][][] train_X = new double[train_N][channel][imageSize[0]][imageSize[1]];
        int[][] train_T = new int[train_N][nOut];
        double[][][][] test_X = new double[test_N][channel][imageSize[0]][imageSize[1]];
        Integer[][] test_T = new Integer[test_N][nOut];
        Integer[][] predicted_T = new Integer[test_N][nOut];

        int epochs = 500;
        double learningRate = 0.1;

        final int minibatchSize = 25;
        int minibatch_N = train_N / minibatchSize;

        double[][][][][] train_X_minibatch = new double[minibatch_N][minibatchSize][channel][imageSize[0]][imageSize[1]];
        int[][][] train_T_minibatch = new int[minibatch_N][minibatchSize][nOut];
        List<Integer> minibatchIndex = new ArrayList<>();
        for (int i = 0; i < train_N; i++) minibatchIndex.add(i);
        Collections.shuffle(minibatchIndex, rng);
        //
        // Create training data and test data for demo.
        //
        for (int pattern = 0; pattern < patterns; pattern++) {
            for (int n = 0; n < train_N_each; n++) {
                int n_ = pattern * train_N_each + n;
                for (int c = 0; c < channel; c++) {
                    for (int i = 0; i < imageSize[0]; i++) {
                        for (int j = 0; j < imageSize[1]; j++) {
                            if ((i < (pattern + 1) * (imageSize[0] / patterns)) && (i >= pattern * imageSize[0] / patterns)) {
                                train_X[n_][c][i][j] = ((int) 128. * rng.nextDouble() + 128.) * RandomGenerator.binomial(1, 1 - pNoise_Training, rng) / 256.;
                            } else {
                                train_X[n_][c][i][j] = 128. * RandomGenerator.binomial(1, pNoise_Training, rng) / 256.;
                            }
                        }
                    }
                }
                for (int i = 0; i < nOut; i++) {
                    if (i == pattern) {
                        train_T[n_][i] = 1;
                    } else {
                        train_T[n_][i] = 0;
                    }
                }
            }
            for (int n = 0; n < test_N_each; n++) {
                int n_ = pattern * test_N_each + n;
                for (int c = 0; c < channel; c++) {
                    for (int i = 0; i < imageSize[0]; i++) {
                        for (int j = 0; j < imageSize[1]; j++) {
                            if ((i < (pattern + 1) * imageSize[0] / patterns) && (i >= pattern * imageSize[0] / patterns)) {
                                test_X[n_][c][i][j] = ((int) 128. * rng.nextDouble() + 128.) * RandomGenerator.binomial(1, 1 - pNoise_Test, rng) / 256.;
                            } else {
                                test_X[n_][c][i][j] = 128. * RandomGenerator.binomial(1, pNoise_Test, rng) / 256.;
                            }
                        }
                    }
                }

                for (int i = 0; i < nOut; i++) {
                    if (i == pattern) {
                        test_T[n_][i] = 1;
                    } else {
                        test_T[n_][i] = 0;
                    }
                }
            }
        }

        // create minibatches
        for (int j = 0; j < minibatchSize; j++) {
            for (int i = 0; i < minibatch_N; i++) {
                train_X_minibatch[i][j] = train_X[minibatchIndex.get(i * minibatchSize + j)];
                train_T_minibatch[i][j] = train_T[minibatchIndex.get(i * minibatchSize + j)];
            }
        }
        //
        // Build Convolutional Neural Networks model
        //
        // construct CNN
        System.out.print("Building the model...");
        ConvolutionalNeuralNetworks classifier = new ConvolutionalNeuralNetworks(imageSize, channel, nKernels, kernelSizes, poolSizes, nHidden, nOut, rng, "ReLU");
        System.out.println("done.");
        // train the model
        System.out.print("Training the model...");
        System.out.println();
        for (int epoch = 0; epoch < epochs; epoch++) {
            if ((epoch + 1) % 50 == 0) {
                System.out.println("\titer = " + (epoch + 1) + " / " + epochs);
            }
            for (int batch = 0; batch < minibatch_N; batch++) {
                classifier.train(train_X_minibatch[batch], train_T_minibatch[batch], minibatchSize, learningRate);
            }
            learningRate *= 0.999;
        }
        System.out.println("done.");
        // test
        for (int i = 0; i < test_N; i++) {
            predicted_T[i] = classifier.predict(test_X[i]);
        }
        //
        // Evaluate the model
        //
        int[][] confusionMatrix = new int[patterns][patterns];
        double accuracy = 0.;
        double[] precision = new double[patterns];
        double[] recall = new double[patterns];
        for (int i = 0; i < test_N; i++) {
            int predicted_ = Arrays.asList(predicted_T[i]).indexOf(1);
            int actual_ = Arrays.asList(test_T[i]).indexOf(1);
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
        System.out.println("CNN model evaluation");
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
CNN model evaluation
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