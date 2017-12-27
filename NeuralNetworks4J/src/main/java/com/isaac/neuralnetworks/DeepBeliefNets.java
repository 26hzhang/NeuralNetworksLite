package com.isaac.neuralnetworks;

import com.isaac.initialization.Activation;
import com.isaac.layers.DenseLayer;
import com.isaac.layers.LogisticLayer;
import com.isaac.layers.RestrictedBoltzmannMachine;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

@SuppressWarnings("unused")
public class DeepBeliefNets {
    private int nIn;
    private int[] hiddenLayerSizes;
    private int nOut;
    private int nLayers;
    private RestrictedBoltzmannMachine[] rbmLayers;
    private DenseLayer[] denseLayers;
    private LogisticLayer outputLayer;
    public Random rng;

    public DeepBeliefNets(int nIn, int[] hiddenLayerSizes, int nOut, Random rng) {
        this.nIn = nIn;
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.nLayers = hiddenLayerSizes.length;
        this.nOut = nOut;
        this.rng = rng == null ? new Random(1234) : rng;
        this.denseLayers = new DenseLayer[nLayers];
        this.rbmLayers = new RestrictedBoltzmannMachine[nLayers];
        // construct multi-layer
        for (int i = 0; i < nLayers; i++) {
            int nIn_;
            if (i == 0) nIn_ = nIn;
            else nIn_ = hiddenLayerSizes[i-1];
            // construct hidden layers with sigmoid function
            // weight matrices and bias vectors will be shared with RBM layers
            denseLayers[i] = new DenseLayer(nIn_, hiddenLayerSizes[i], null, null, rng, Activation.Sigmoid);
            // construct RBM layers
            rbmLayers[i] = new RestrictedBoltzmannMachine(nIn_, hiddenLayerSizes[i], denseLayers[i].getW(),
                    denseLayers[i].getB(), null, rng, null);
        }
        // logistic regression layer for output
        outputLayer = new LogisticLayer(hiddenLayerSizes[nLayers-1], nOut);
    }

    public void pretrain(int[][][] X, int minibatchSize, int minibatch_N, int epochs, double learningRate, int k) {
        for (int layer = 0; layer < nLayers; layer++) {  // pre-train layer-wise
            for (int epoch = 0; epoch < epochs; epoch++) {
                for (int batch = 0; batch < minibatch_N; batch++) {
                    int[][] X_ = new int[minibatchSize][nIn];
                    int[][] prevLayerX_;
                    // Set input data for current layer
                    if (layer == 0) { X_ = X[batch]; }
                    else {
                        prevLayerX_ = X_;
                        X_ = new int[minibatchSize][hiddenLayerSizes[layer-1]];
                        for (int i = 0; i < minibatchSize; i++) { X_[i] = denseLayers[layer-1].outputBinomial(prevLayerX_[i], rng); }
                    }
                    rbmLayers[layer].contrastiveDivergence(X_, minibatchSize, learningRate, k);
                }
            }
        }
    }

    public void finetune(double[][] X, int[][] T, int minibatchSize, double learningRate) {
        List<double[][]> layerInputs = new ArrayList<>(nLayers + 1);
        layerInputs.add(X);
        double[][] Z = new double[0][0];
        double[][] dY;
        // forward hidden layers
        for (int layer = 0; layer < nLayers; layer++) {
            double[] x_;  // layer input
            double[][] Z_ = new double[minibatchSize][hiddenLayerSizes[layer]];
            for (int n = 0; n < minibatchSize; n++) {
                if (layer == 0) x_ = X[n];
                else x_ = Z[n];
                Z_[n] = denseLayers[layer].forward(x_);
            }
            Z = Z_;
            layerInputs.add(Z.clone());
        }
        // forward & backward output layer
        dY = outputLayer.train(Z, T, minibatchSize, learningRate);
        // backward hidden layers
        double[][] Wprev;
        double[][] dZ = new double[0][0];
        for (int layer = nLayers - 1; layer >= 0; layer--) {
            if (layer == nLayers - 1) Wprev = outputLayer.getW();
            else {
                Wprev = denseLayers[layer+1].getW();
                dY = dZ.clone();
            }
            dZ = denseLayers[layer].backward(layerInputs.get(layer), layerInputs.get(layer+1), dY, Wprev, minibatchSize,
                    learningRate);
        }
    }

    public Integer[] predict(double[] x) {
        double[] z = new double[0];
        for (int layer = 0; layer < nLayers; layer++) {
            double[] x_;
            if (layer == 0) x_ = x;
            else x_ = z.clone();
            z = denseLayers[layer].forward(x_);
        }
        return outputLayer.predict(z);
    }

    /** Getters and Setters */
    public int getnIn() { return nIn; }
    public void setnIn(int nIn) { this.nIn = nIn; }
    public int[] getHiddenLayerSizes() { return hiddenLayerSizes; }
    public void setHiddenLayerSizes(int[] hiddenLayerSizes) { this.hiddenLayerSizes = hiddenLayerSizes; }
    public int getnOut() { return nOut; }
    public void setnOut(int nOut) { this.nOut = nOut; }
    public int getnLayers() { return nLayers; }
    public void setnLayers(int nLayers) { this.nLayers = nLayers; }
    public Random getRng() { return rng; }
    public void setRng(Random rng) { this.rng = rng; }
}
