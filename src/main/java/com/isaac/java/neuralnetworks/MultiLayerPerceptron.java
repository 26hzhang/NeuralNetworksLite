package com.isaac.java.neuralnetworks;

import java.util.Random;

public class MultiLayerPerceptron {
	public int nIn;
	public int nHidden;
	public int nOut;
	public HiddenLayer hiddenLayer;
	public LogisticRegression logisticLayer;
	public Random rng;

	public MultiLayerPerceptron(int nIn, int nHidden, int nOut, Random rng) {
		this.nIn = nIn;
		this.nHidden = nHidden;
		this.nOut = nOut;
		this.rng = rng == null ? new Random(1234) : rng;
		// construct hidden layer with tanh as activation function
		hiddenLayer = new HiddenLayer(nIn, nHidden, null, null, rng, "tanh"); // sigmoid or tanh
		// construct output layer i.e. multi-class logistic layer
		logisticLayer = new LogisticRegression(nHidden, nOut);
	}

	public void train(double[][] X, int T[][], int minibatchSize, double learningRate) {
		double[][] Z = new double[minibatchSize][nIn]; // outputs of hidden layer (= inputs of output layer)
		double[][] dY;
		// forward hidden layer
		for (int n = 0; n < minibatchSize; n++) {
			Z[n] = hiddenLayer.forward(X[n]); // activate input units
		}
		// forward & backward output layer
		dY = logisticLayer.train(Z, T, minibatchSize, learningRate);
		// backward hidden layer (backpropagate)
		hiddenLayer.backward(X, Z, dY, logisticLayer.W, minibatchSize, learningRate);
	}

	public Integer[] predict(double[] x) {
		double[] z = hiddenLayer.output(x);
		return logisticLayer.predict(z);
	}
}
