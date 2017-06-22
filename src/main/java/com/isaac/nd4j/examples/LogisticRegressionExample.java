package com.isaac.nd4j.examples;

import com.isaac.nd4j.initialization.Activation;
import com.isaac.nd4j.initialization.Weight;
import com.isaac.nd4j.layers.OutputLayer;
import com.isaac.nd4j.utils.GaussianDistribution;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class LogisticRegressionExample {

	public static void main(String[] args) {
		final Random rng = new Random(1234); // seed random
		/*
		 * Declare variables and constants
		 */
		final int patterns = 3; // number of classes
		final int train_N = 400 * patterns;
		final int test_N = 60 * patterns;
		final int nIn = 2;
		final int nOut = 3;

		INDArray train_X = Nd4j.create(new double[train_N * nIn], new int[] {train_N, nIn});
		INDArray train_T = Nd4j.create(new double[train_N * nOut], new int[] {train_N, nOut});

		INDArray test_X = Nd4j.create(new double[test_N * nIn], new int[] {test_N, nIn});
		INDArray test_T = Nd4j.create(new double[test_N * nOut], new int[] {test_N, nOut});

		int epochs = 2000; /* iteration times */
		double learningRate = 0.2;

		int minibatchSize = 50; // number of data in each minibatch
		int minibatch_N = train_N / minibatchSize; // number of minibatches

		List<INDArray> train_X_minibatch = new ArrayList<>();
		List<INDArray> train_T_minibatch = new ArrayList<>();

		List<Integer> minibatchIndex = new ArrayList<>(); // data index for minibatch to apply SGD
		for (int i = 0; i < train_N; i++) {
			minibatchIndex.add(i);
		}
		Collections.shuffle(minibatchIndex, rng); // shuffle data index for SGD

		/*
		 * Training data for demo
		 * class 1 : x1 ~ N(-2.0, 1.0), y1 ~ N(+2.0, 1.0)
		 * class 2 : x2 ~ N(+2.0, 1.0), y2 ~ N(-2.0, 1.0)
		 * class 3 : x3 ~ N( 0.0, 1.0), y3 ~ N( 0.0, 1.0)
		 */
		GaussianDistribution g1 = new GaussianDistribution(-2.0, 1.0, rng);
		GaussianDistribution g2 = new GaussianDistribution(2.0, 1.0, rng);
		GaussianDistribution g3 = new GaussianDistribution(0.0, 1.0, rng);
		// data set in class 1
		for (int i = 0; i < train_N / patterns - 1; i++) {
			train_X.put(i, 0, Nd4j.scalar(g1.random()));
			train_X.put(i, 1, Nd4j.scalar(g2.random()));
			train_T.putRow(i, Nd4j.create(new double[] {1.0, 0.0, 0.0}));
		}
		for (int i = 0; i < test_N / patterns - 1; i++) {
			test_X.put(i, 0, Nd4j.scalar(g1.random()));
			test_X.put(i, 1, Nd4j.scalar(g2.random()));
			test_T.putRow(i, Nd4j.create(new double[] {1.0, 0.0, 0.0}));
		}

		// data set in class 2
		for (int i = train_N / patterns - 1; i < train_N / patterns * 2 - 1; i++) {
			train_X.put(i, 0, Nd4j.scalar(g2.random()));
			train_X.put(i, 1, Nd4j.scalar(g1.random()));
			train_T.putRow(i, Nd4j.create(new double[] {0.0, 1.0, 0.0}));
		}
		for (int i = test_N / patterns - 1; i < test_N / patterns * 2 - 1; i++) {
			test_X.put(i, 0, Nd4j.scalar(g2.random()));
			test_X.put(i, 1, Nd4j.scalar(g1.random()));
			test_T.putRow(i, Nd4j.create(new double[] {0.0, 1.0, 0.0}));
		}

		// data set in class 3
		for (int i = train_N / patterns * 2 - 1; i < train_N; i++) {
			train_X.put(i, 0, Nd4j.scalar(g3.random()));
			train_X.put(i, 1, Nd4j.scalar(g3.random()));
			train_T.putRow(i, Nd4j.create(new double[] {0.0, 0.0, 1.0}));
		}
		for (int i = test_N / patterns * 2 - 1; i < test_N; i++) {
			test_X.put(i, 0, Nd4j.scalar(g3.random()));
			test_X.put(i, 1, Nd4j.scalar(g3.random()));
			test_T.putRow(i, Nd4j.create(new double[] {0.0, 0.0, 1.0}));
		}

		// create minibatches with training data
		for (int i = 0; i < minibatch_N; i++) {
			INDArray trainX = Nd4j.create(new double[minibatchSize * nIn], new int[] {minibatchSize, nIn});
			INDArray trainT = Nd4j.create(new double[minibatchSize * nOut], new int[] {minibatchSize, nOut});
			for (int j = 0; j < minibatchSize; j++) {
				trainX.putRow(j, train_X.getRow(minibatchIndex.get(i * minibatchSize + j)));
				trainT.putRow(j, train_T.getRow(minibatchIndex.get(i * minibatchSize + j)));
			}
			train_X_minibatch.add(trainX);
			train_T_minibatch.add(trainT);
		}

		/*
		 * Build Logistic Regression model
		 */
		OutputLayer classifier = new OutputLayer(nIn, nOut, Weight.ZERO, Activation.SoftMax, null);

		// train
		for (int epoch = 0; epoch < epochs; epoch++) {
			for (int batch = 0; batch < minibatch_N; batch++) {
				classifier.train(train_X_minibatch.get(batch), train_T_minibatch.get(batch), minibatchSize,
						learningRate);
			}
			learningRate *= 0.95;
		}
		// test
		INDArray predicted_T = classifier.predict(test_X);

		/*
		 * Evaluate the model
		 */
		int[][] confusionMatrix = new int[patterns][patterns];
		double accuracy = 0.0;
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

		System.out.println("------------------------------------");
		System.out.println("Logistic Regression model evaluation");
		System.out.println("------------------------------------");
		System.out.println(String.format("Accuracy: %.1f\n", accuracy * 100));
		System.out.println("Precision:");
		for (int i = 0; i < patterns; i++) {
			System.out.printf("class %d: %.1f\n", i + 1, precision[i] * 100);
		}
		System.out.println("Recall:");
		for (int i = 0; i < patterns; i++) {
			System.out.printf("class %d: %.1f\n", i + 1, recall[i] * 100);
		}
	}

}
/*
Logistic Regression model evaluation
------------------------------------
Accuracy: 91.1

Precision:
class 1: 95.0
class 2: 91.5
class 3: 86.9
Recall:
class 1: 96.6
class 2: 90.0
class 3: 86.9
 */