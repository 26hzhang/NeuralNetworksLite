package com.isaac.neuralnetworks;

import com.isaac.utils.ActivationFunction;

public class Perceptron {

	public int nIn; // dimensions of input data
	public double[] w; // weight vector of perceptrons
	public double learningRate;
	
	public Perceptron(int nIn) {
		this.nIn = nIn;
		this.w = new double[nIn];
	}
	
	public int train(double[] x, int t, double learningRate) {
		int classified = 0;
		double c = 0;
		// check if the data is classified correctly
		for (int i = 0; i < nIn; i++) {
			c += w[i] * x[i] * t;
		}
		// apply steepest descent method if the data is wrongly classified
		if (c > 0) {
			classified = 1;
		} else {
			for (int i = 0; i < nIn; i++) {
				w[i] += learningRate * x[i] * t;
			}
		}
		return classified;
	}
	
	public int predict(double[] x) {
		double preActivation = 0.0;
		for (int i = 0; i < nIn; i++) {
			preActivation += w[i] * x[i];
		}
		return ActivationFunction.step(preActivation);
	}
}
