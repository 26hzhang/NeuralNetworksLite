package com.isaac.nns.examples.basic;

import com.isaac.nns.layers.basic.Dropout;
import com.isaac.nns.utils.RandomGenerator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class DropoutExample {

	public static void main(String[] args) {
		final Random rng = new Random(123);
       	//
        // Declare variables and constants
        //
        int train_N_each = 300;        // for demo
        int test_N_each = 50;          // for demo
        int nIn_each = 20;             // for demo
        double pNoise_Training = 0.2;  // for demo
        double pNoise_Test = 0.25;     // for demo

        final int patterns = 3;
        final int train_N = train_N_each * patterns;
        final int test_N = test_N_each * patterns;
        final int nIn = nIn_each * patterns;
        final int nOut = patterns;

        int[] hiddenLayerSizes = {100, 80};
        double pDropout = 0.5;
        double[][] train_X = new double[train_N][nIn];
        int[][] train_T = new int[train_N][nOut];
        double[][] test_X = new double[test_N][nIn];
        Integer[][] test_T = new Integer[test_N][nOut];
        Integer[][] predicted_T = new Integer[test_N][nOut];

        int epochs = 5000;
        double learningRate = 0.2;
        int minibatchSize = 50;
        final int minibatch_N = train_N / minibatchSize;

        double[][][] train_X_minibatch = new double[minibatch_N][minibatchSize][nIn];
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
                for (int i = 0; i < nIn; i++) {
                    if ( (n_ >= train_N_each * pattern && n_ < train_N_each * (pattern + 1) ) &&
                            (i >= nIn_each * pattern && i < nIn_each * (pattern + 1)) ) {
                        train_X[n_][i] = RandomGenerator.binomial(1, 1 - pNoise_Training, rng) *
                        		rng.nextDouble() * .5 + .5;
                    } else {
                        train_X[n_][i] = RandomGenerator.binomial(1, pNoise_Training, rng) * 
                        		rng.nextDouble() * .5 + .5;
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
                for (int i = 0; i < nIn; i++) {
                    if ( (n_ >= test_N_each * pattern && n_ < test_N_each * (pattern + 1) ) &&
                            (i >= nIn_each * pattern && i < nIn_each * (pattern + 1)) ) {
                        test_X[n_][i] = (double) RandomGenerator.binomial(1, 1 - pNoise_Test, rng) * 
                        		rng.nextDouble() * .5 + .5;
                    } else {
                        test_X[n_][i] = (double) RandomGenerator.binomial(1, pNoise_Test, rng) * 
                        		rng.nextDouble() * .5 + .5;
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
        // Build Dropout model
        //
        // construct Dropout
        System.out.print("Building the model...");
        Dropout classifier = new Dropout(nIn, hiddenLayerSizes, nOut, rng, "ReLU");
        System.out.println("done.");

        // train the model
        System.out.print("Training the model...");
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int batch = 0; batch < minibatch_N; batch++) {
                classifier.train(train_X_minibatch[batch], train_T_minibatch[batch], minibatchSize, learningRate, pDropout);
            }
            learningRate *= 0.999;
        }
        System.out.println("done.");
        // adjust the weight for testing
        System.out.print("Optimizing weights before testing...");
        classifier.pretest(pDropout);
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

        System.out.println("------------------------");
        System.out.println("Dropout model evaluation");
        System.out.println("------------------------");
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
Dropout model evaluation
------------------------
Accuracy: 97.3 %
Precision:
 class 1: 96.1 %
 class 2: 96.0 %
 class 3: 100.0 %
Recall:
 class 1: 98.0 %
 class 2: 96.0 %
 class 3: 98.0 %
 */