package com.isaac.nns.neuralnetworks.basic;

import com.isaac.nns.initialization.ActivationFunction;

public class LogisticRegression {
	public int nIn;
	public int nOut;
	public double[][] W;
	public double[] b;

	public LogisticRegression(int nIn, int nOut) {
		this.nIn = nIn;
		this.nOut = nOut;
		W = new double[nOut][nIn];
		b = new double[nOut];
	}
	
	public double[][] train(double[][] X, int T[][], int minibatchSize, double learningRate) {
		double[][] grad_W = new double[nOut][nIn];
		double[] grad_b = new double[nOut];

		double[][] dY = new double[minibatchSize][nOut];
		// train with SGD
		// 1. calculate gradient of W, b
		for (int n = 0; n < minibatchSize; n++) {
			double[] predicted_Y_ = output(X[n]);
			for (int j = 0; j < nOut; j++) {
				dY[n][j] = predicted_Y_[j] - T[n][j];
				for (int i = 0; i < nIn; i++) {
					grad_W[j][i] += dY[n][j] * X[n][i];
				}
				grad_b[j] += dY[n][j];
			}
		}
		// 2. update params
		for (int j = 0; j < nOut; j++) {
			for (int i = 0; i < nIn; i++) {
				W[j][i] -= learningRate * grad_W[j][i] / minibatchSize;
			}
			b[j] -= learningRate * grad_b[j] / minibatchSize;
		}
		return dY;
	}

	public double[] output(double[] x) {
		double[] preActivation = new double[nOut];
		for (int j = 0; j < nOut; j++) {
			for (int i = 0; i < nIn; i++) {
				preActivation[j] += W[j][i] * x[i];
			}
			preActivation[j] += b[j]; // linear output
		}
		return ActivationFunction.softmax(preActivation, nOut);
	}

	public Integer[] predict(double[] x) {
		double[] y = output(x); // activate input data through learned networks
		Integer[] t = new Integer[nOut]; // output is the probability, so cast it to label
		int argmax = -1;
		double max = 0.0;
		for (int i = 0; i < nOut; i++) {
			if (max < y[i]) {
				max = y[i];
				argmax = i;
			}
		}
		for (int i = 0; i < nOut; i++) {
			if (i == argmax) {
				t[i] = 1;
			} else {
				t[i] = 0;
			}
		}
		return t;
	}
}
