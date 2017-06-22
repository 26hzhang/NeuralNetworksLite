package com.isaac.java.neuralnetworks;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.isaac.java.utils.RandomGenerator;

public class Dropout {
	public int nIn;
	public int nOut;
	public int[] hiddenLayerSizes;
	public int nLayers;
	public HiddenLayer[] hiddenLayers;
	public LogisticRegression logisticLayer;
	public Random rng;
	
	public Dropout(int nIn, int[] hiddenLayerSizes, int nOut, Random rng, String activation) {
		if (rng == null)
			rng = new Random(1234); // seed random
		if (activation == null)
			activation = "ReLU";
		this.nIn = nIn;
		this.nOut = nOut;
		this.hiddenLayerSizes = hiddenLayerSizes;
		this.nLayers = hiddenLayerSizes.length;
		this.hiddenLayers = new HiddenLayer[nLayers];
		this.rng = rng;
		// construct multi-layer
		for (int i = 0; i < nLayers; i++) {
			int nIn_;
			if (i == 0)
				nIn_ = nIn;
			else
				nIn_ = hiddenLayerSizes[i - 1];
			// construct hidden layer
			hiddenLayers[i] = new HiddenLayer(nIn_, hiddenLayerSizes[i], null, null, rng, activation);
		}
		// construct logistic layer
		logisticLayer = new LogisticRegression(hiddenLayerSizes[nLayers - 1], nOut);
	}
	
	public void train(double[][] X, int[][] T, int minibatchSize, double learningRate, double pDrouput) {
		// since we need some layer inputs when calculating the back-propagation errors,
		// define layerInputs to cache their respective input values
		List<double[][]> layerInputs = new ArrayList<double[][]>(nLayers + 1);
		layerInputs.add(X); // here the X is original training data
		// cache dropout masks for each layer for back-propagation
		List<int[][]> dropoutMasks = new ArrayList<int[][]>(nLayers);
		double[][] Z = new double[0][0];
        double[][] D; // delta
        // forward hidden layers
        for (int layer = 0; layer < nLayers; layer++) {
        	double[] x_; // layer input
        	double[][] Z_ = new double[minibatchSize][hiddenLayerSizes[layer]];
        	int[][] mask_ = new int[minibatchSize][hiddenLayerSizes[layer]];
        	for (int n = 0; n < minibatchSize; n++) {
        		if (layer == 0) {
        			x_ = X[n];
        		} else {
        			x_ = Z[n];
        		}
        		Z_[n] = hiddenLayers[layer].forward(x_);
        		mask_[n] = dropout(Z_[n], pDrouput); // apply dropout mask to units
        	}
        	Z = Z_;
        	layerInputs.add(Z.clone());
        	dropoutMasks.add(mask_);
        }
        // After forward propagation through the hidden layers, training data is 
        // forward propagated in the output layer of the logistic regression.
        // Then, the deltas of each layer are going back through the network.
        // Here, we apply the cached masks to the delta so that its values are 
        // back-propagated in the same network
        // forward & backward output layer
        D = logisticLayer.train(Z, T, minibatchSize, learningRate);
        // backward hidden layers
        for (int layer = nLayers - 1; layer >= 0; layer--) {
        	double[][] Wprev_;
        	if (layer == nLayers - 1) {
        		Wprev_ = logisticLayer.W;
        	} else {
        		Wprev_= hiddenLayers[layer + 1].W;
        	}
        	// apply mask to delta as well
        	for (int n = 0; n < minibatchSize; n++) {
        		int[] mask_ = dropoutMasks.get(layer)[n];
        		for (int j = 0; j < D[n].length; j++) {
        			D[n][j] *= mask_[j];
        		}
        	}
        	D = hiddenLayers[layer].backward(layerInputs.get(layer), layerInputs.get(layer + 1), D, Wprev_, 
        			minibatchSize, learningRate);
        }
	}
	
	// this function returns the values following the Bernoulli distribution
	public int[] dropout(double[] z, double p) {
		int size = z.length;
		int[] mask = new int[size];
		for (int i = 0; i < size; i++) {
			mask[i] = RandomGenerator.binomial(1, 1 - p, rng);
			z[i] *= mask[i]; // apply mask
		}
		return mask;
	}
	
	// Before applying the test data to tuned model, we need to configure the weights of the network.
	// Dropout masks cannot be simply applied to the test data, because when masked, the shape of each
	// network will be differentiated, and this may returns different results because a certain unit
	// may have a significant effect on certain features. Instead, we need to smooth the weights of the
	// network. Below, all the weights are multiplied by the probability of non-dropout.
	public void pretest(double pDropout) {
        for (int layer = 0; layer < nLayers; layer++) {
            int nIn_, nOut_;
            if (layer == 0) {
                nIn_ = nIn;
            } else {
                nIn_ = hiddenLayerSizes[layer];
            }
            if (layer == nLayers - 1) {
                nOut_ = nOut;
            } else {
                nOut_ = hiddenLayerSizes[layer+1];
            }
            for (int j = 0; j < nOut_; j++) {
                for (int i = 0; i < nIn_; i++) {
                    hiddenLayers[layer].W[j][i] *= 1 - pDropout;
                }
            }
        }
    }

    public Integer[] predict(double[] x) {
        double[] z = new double[0];
        for (int layer = 0; layer < nLayers; layer++) {
            double[] x_;
            if (layer == 0) {
                x_ = x;
            } else {
                x_ = z.clone();
            }
            z = hiddenLayers[layer].forward(x_);
        }
        return logisticLayer.predict(z);
    }
}
