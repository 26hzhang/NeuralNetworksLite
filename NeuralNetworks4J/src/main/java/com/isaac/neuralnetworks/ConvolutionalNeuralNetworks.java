package com.isaac.neuralnetworks;


import com.isaac.initialization.Activation;
import com.isaac.layers.ConvolutionPoolingLayer;
import com.isaac.layers.DenseLayer;
import com.isaac.layers.LogisticLayer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

@SuppressWarnings("unused")
public class ConvolutionalNeuralNetworks {
    private int[] nKernels;
    private int[][] kernelSizes;
    private int[][] poolSizes;
    private int nHidden;
    private int nOut;
    private ConvolutionPoolingLayer[] convPoolLayers;
    private int[][] convolvedSizes;
    private int[][] pooledSizes;
    private int flattenedSize;
    private DenseLayer denseLayer;
    private LogisticLayer outputLayer;
    private Random rng;

    public ConvolutionalNeuralNetworks(int[] imageSize, int channel, int[] nKernels, int[][] kernelSizes,
                                       int[][] poolSizes, int nHidden, int nOut, Random rng, Activation activationMethod) {
        this.nKernels = nKernels;
        this.kernelSizes = kernelSizes;
        this.poolSizes = poolSizes;
        this.nHidden = nHidden;
        this.nOut = nOut;
        this.rng = rng == null ? new Random(1234) : rng;
        convPoolLayers = new ConvolutionPoolingLayer[nKernels.length];
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
            convPoolLayers[i] = new ConvolutionPoolingLayer(size_, channel_, nKernels[i], kernelSizes[i], poolSizes[i],
                    convolvedSizes[i], pooledSizes[i], rng, activationMethod);
        }
        // build MLP
        flattenedSize = nKernels[nKernels.length-1] * pooledSizes[pooledSizes.length-1][0] * pooledSizes[pooledSizes.length-1][1];
        // construct hidden layer
        denseLayer = new DenseLayer(flattenedSize, nHidden, null, null, rng, activationMethod);
        // construct output layer
        outputLayer = new LogisticLayer(nHidden, nOut);
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
                z_ = convPoolLayers[i].forward(z_, preActivated_X.get(i)[n], activated_X.get(i)[n]);
                downsampled_X.get(i+1)[n] = z_.clone();
            }
            // flatten output to make it input for fully connected MLP
            double[] x_ = this.flatten(z_);
            flattened_X[n] = x_.clone();
            // forward hidden layer
            Z[n] = denseLayer.forward(x_);
        }
        // forward & backward output layer
        dY = outputLayer.train(Z, T, minibatchSize, learningRate);
        // backward hidden layer
        dZ = denseLayer.backward(flattened_X, Z, dY, outputLayer.getW(), minibatchSize, learningRate);
        // back-propagate delta to input layer
        for (int n = 0; n < minibatchSize; n++) {
            for (int i = 0; i < flattenedSize; i++) {
                for (int j = 0; j < nHidden; j++) { dX_flatten[n][i] += denseLayer.getW()[j][i] * dZ[n][j]; }
            }
            dX[n] = unflatten(dX_flatten[n]);  // un-flatten delta
        }
        // backward convolution + pooling layers
        dC = dX.clone();
        for (int i = nKernels.length-1; i >= 0; i--) {
            dC = convPoolLayers[i].backward(downsampled_X.get(i), preActivated_X.get(i), activated_X.get(i),
                    downsampled_X.get(i+1), dC, minibatchSize, learningRate);
        }
    }

    private double[] flatten(double[][][] z) {
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

    private double[][][] unflatten(double[] x) {
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
            z = convPoolLayers[i].forward(z, preActivated.get(i), activated.get(i));
        }
        // forward MLP
        return outputLayer.predict(denseLayer.forward(this.flatten(z)));
    }

    /** Getters and Setters */
    public int[] getnKernels() { return nKernels; }
    public void setnKernels(int[] nKernels) { this.nKernels = nKernels; }
    public int[][] getKernelSizes() { return kernelSizes; }
    public void setKernelSizes(int[][] kernelSizes) { this.kernelSizes = kernelSizes; }
    public int[][] getPoolSizes() { return poolSizes; }
    public void setPoolSizes(int[][] poolSizes) { this.poolSizes = poolSizes; }
    public int getnHidden() { return nHidden; }
    public void setnHidden(int nHidden) { this.nHidden = nHidden; }
    public int getnOut() { return nOut; }
    public void setnOut(int nOut) { this.nOut = nOut; }
    public int[][] getConvolvedSizes() { return convolvedSizes; }
    public void setConvolvedSizes(int[][] convolvedSizes) { this.convolvedSizes = convolvedSizes; }
    public int[][] getPooledSizes() { return pooledSizes; }
    public void setPooledSizes(int[][] pooledSizes) { this.pooledSizes = pooledSizes; }
    public int getFlattenedSize() { return flattenedSize; }
    public void setFlattenedSize(int flattenedSize) { this.flattenedSize = flattenedSize; }
    public Random getRng() { return rng; }
    public void setRng(Random rng) { this.rng = rng; }
}
