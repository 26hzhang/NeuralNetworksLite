package com.isaac.java.neuralnetworks;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ConvolutionalNeuralNetworks {
	public int[] nKernels;
    public int[][] kernelSizes;
    public int[][] poolSizes;
    public int nHidden;
    public int nOut;

    public ConvolutionPoolingLayer[] convpoolLayers;
    public int[][] convolvedSizes;
    public int[][] pooledSizes;
    public int flattenedSize;
    public HiddenLayer hiddenLayer;
    public LogisticRegression logisticLayer;
    public Random rng;

    public ConvolutionalNeuralNetworks(int[] imageSize, int channel, int[] nKernels, int[][] kernelSizes, 
    		int[][] poolSizes, int nHidden, int nOut, Random rng, String activation) {
        if (rng == null) 
        	rng = new Random(1234);
        this.nKernels = nKernels;
        this.kernelSizes = kernelSizes;
        this.poolSizes = poolSizes;
        this.nHidden = nHidden;
        this.nOut = nOut;
        this.rng = rng;
        convpoolLayers = new ConvolutionPoolingLayer[nKernels.length];
        convolvedSizes = new int[nKernels.length][imageSize.length];
        pooledSizes = new int[nKernels.length][imageSize.length];
        // construct convolution + pooling layers
        for (int i = 0; i < nKernels.length; i++) {
            int[] size_;
            int channel_;
            if (i == 0) {
                size_ = new int[]{imageSize[0], imageSize[1]};
                channel_ = channel;
            } else {
                size_ = new int[]{pooledSizes[i-1][0], pooledSizes[i-1][1]};
                channel_ = nKernels[i-1];
            }
            convolvedSizes[i] = new int[]{size_[0] - kernelSizes[i][0] + 1, size_[1] - kernelSizes[i][1] + 1};
            pooledSizes[i] = new int[]{convolvedSizes[i][0] / poolSizes[i][0], convolvedSizes[i][1] / poolSizes[i][0]};
            convpoolLayers[i] = new ConvolutionPoolingLayer(size_, channel_, nKernels[i], kernelSizes[i], poolSizes[i], convolvedSizes[i], pooledSizes[i], rng, activation);
        }
        // build MLP
        flattenedSize = nKernels[nKernels.length-1] * pooledSizes[pooledSizes.length-1][0] * pooledSizes[pooledSizes.length-1][1];
        // construct hidden layer
        hiddenLayer = new HiddenLayer(flattenedSize, nHidden, null, null, rng, activation);
        // construct output layer
        logisticLayer = new LogisticRegression(nHidden, nOut);
    }


    public void train(double[][][][] X, int[][] T, int minibatchSize, double learningRate) {
        // cache pre-activated, activated, and downsampled inputs of each convolution + pooling layer for backpropagation
        List<double[][][][]> preActivated_X = new ArrayList<>(nKernels.length);
        List<double[][][][]> activated_X = new ArrayList<>(nKernels.length);
        List<double[][][][]> downsampled_X = new ArrayList<>(nKernels.length+1);  // +1 for input X
        downsampled_X.add(X);
        for (int i = 0; i < nKernels.length; i++) {
            preActivated_X.add(new double[minibatchSize][nKernels[i]][convolvedSizes[i][0]][convolvedSizes[i][1]]);
            activated_X.add(new double[minibatchSize][nKernels[i]][convolvedSizes[i][0]][convolvedSizes[i][1]]);
            downsampled_X.add(new double[minibatchSize][nKernels[i]][convolvedSizes[i][0]][convolvedSizes[i][1]]);
        }
        double[][] flattened_X = new double[minibatchSize][flattenedSize];  // cache flattened inputs
        double[][] Z = new double[minibatchSize][nHidden];  // cache outputs of hidden layer
        double[][] dY;  // delta of output layer
        double[][] dZ;  // delta of hidden layer
        double[][] dX_flatten = new double[minibatchSize][flattenedSize];  // delta of input layer
        double[][][][] dX = new double[minibatchSize][nKernels[nKernels.length-1]][pooledSizes[pooledSizes.length-1][0]][pooledSizes[pooledSizes.length-1][1]];
        double[][][][] dC;
        // train with minibatch
        for (int n = 0; n < minibatchSize; n++) {
            // forward convolution + pooling layers
            double[][][] z_ = X[n].clone();
            for (int i = 0; i < nKernels.length; i++) {
                z_ = convpoolLayers[i].forward(z_, preActivated_X.get(i)[n], activated_X.get(i)[n]);
                downsampled_X.get(i+1)[n] = z_.clone();
            }
            // flatten output to make it input for fully connected MLP
            double[] x_ = this.flatten(z_);
            flattened_X[n] = x_.clone();
            // forward hidden layer
            Z[n] = hiddenLayer.forward(x_);
        }
        // forward & backward output layer
        dY = logisticLayer.train(Z, T, minibatchSize, learningRate);
        // backward hidden layer
        dZ = hiddenLayer.backward(flattened_X, Z, dY, logisticLayer.W, minibatchSize, learningRate);
        // backpropagate delta to input layer
        for (int n = 0; n < minibatchSize; n++) {
            for (int i = 0; i < flattenedSize; i++) {
                for (int j = 0; j < nHidden; j++) {
                    dX_flatten[n][i] += hiddenLayer.W[j][i] * dZ[n][j];
                }
            }
            dX[n] = unflatten(dX_flatten[n]);  // unflatten delta
        }
        // backward convolution + pooling layers
        dC = dX.clone();
        for (int i = nKernels.length-1; i >= 0; i--) {
            dC = convpoolLayers[i].backward(downsampled_X.get(i), preActivated_X.get(i), activated_X.get(i), downsampled_X.get(i+1), dC, minibatchSize, learningRate);
        }
    }

    public double[] flatten(double[][][] z) {
        double[] x = new double[flattenedSize];
        int index = 0;
        for (int k = 0; k < nKernels[nKernels.length-1]; k++) {
            for (int i = 0; i < pooledSizes[pooledSizes.length-1][0]; i++) {
                for (int j = 0; j < pooledSizes[pooledSizes.length-1][1]; j++) {
                    x[index] = z[k][i][j];
                    index += 1;
                }
            }
        }
        return x;
    }

    public double[][][] unflatten(double[] x) {
        double[][][] z = new double[nKernels[nKernels.length-1]][pooledSizes[pooledSizes.length-1][0]][pooledSizes[pooledSizes.length-1][1]];
        int index = 0;
        for (int k = 0; k < z.length; k++) {
            for (int i = 0; i < z[0].length; i++) {
                for (int j = 0; j < z[0][0].length; j++) {
                    z[k][i][j] = x[index];
                    index += 1;
                }
            }
        }
        return z;
    }

    public Integer[] predict(double[][][] x) {
        List<double[][][]> preActivated = new ArrayList<>(nKernels.length);
        List<double[][][]> activated = new ArrayList<>(nKernels.length);
        for (int i = 0; i < nKernels.length; i++) {
            preActivated.add(new double[nKernels[i]][convolvedSizes[i][0]][convolvedSizes[i][1]]);
            activated.add(new double[nKernels[i]][convolvedSizes[i][0]][convolvedSizes[i][1]]);
        }
        // forward convolution + pooling layers
        double[][][] z = x.clone();
        for (int i = 0; i < nKernels.length; i++) {
            z = convpoolLayers[i].forward(z, preActivated.get(i), activated.get(i));
        }
        // forward MLP
        return logisticLayer.predict(hiddenLayer.forward(this.flatten(z)));
    }
}
