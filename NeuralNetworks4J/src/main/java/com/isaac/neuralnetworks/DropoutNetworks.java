package com.isaac.neuralnetworks;

import com.isaac.initialization.Activation;
import com.isaac.layers.DenseLayer;
import com.isaac.layers.LogisticLayer;
import com.isaac.utils.RandomGenerator;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

@SuppressWarnings({"Duplicates", "unused"})
public class DropoutNetworks {
    private int nIn;
    private int nOut;
    private int[] hiddenLayerSizes;
    private int nLayers;
    private DenseLayer[] hiddenLayers;
    private LogisticLayer outputLayer;
    private Random rng;

    public DropoutNetworks (int nIn, int[] hiddenLayerSizes, int nOut, Random rng, Activation activationMethod) {
        this.nIn = nIn;
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.nLayers = hiddenLayerSizes.length;
        this.nOut = nOut;
        this.rng = rng == null ? new Random(1234) : rng;
        activationMethod = activationMethod == null ? Activation.Tanh : activationMethod;
        this.hiddenLayers = new DenseLayer[nLayers];
        for (int i = 0; i < nLayers; i++) {
            int nIn_;
            if (i == 0) nIn_ = nIn;
            else nIn_ = hiddenLayerSizes[i - 1];
            hiddenLayers[i] = new DenseLayer(nIn_, hiddenLayerSizes[i], null, null, rng, activationMethod);
        }
        outputLayer = new LogisticLayer(hiddenLayerSizes[nLayers - 1], nOut);
    }

    public void train(double[][] X, int[][] T, int minibatchSize, double learningRate, double pDrouput) {
        // since we need some layer inputs when calculating the back-propagation errors,
        // define layerInputs to cache their respective input values
        List<double[][]> layerInputs = new ArrayList<>(nLayers + 1);
        layerInputs.add(X); // here the X is original training data
        // cache dropout masks for each layer for back-propagation
        List<int[][]> dropoutMasks = new ArrayList<>(nLayers);
        double[][] Z = new double[0][0];
        double[][] D; // delta
        // forward hidden layers
        for (int layer = 0; layer < nLayers; layer++) {
            double[] x_; // layer input
            double[][] Z_ = new double[minibatchSize][hiddenLayerSizes[layer]];
            int[][] mask_ = new int[minibatchSize][hiddenLayerSizes[layer]];
            for (int n = 0; n < minibatchSize; n++) {
                if (layer == 0) { x_ = X[n]; }
                else { x_ = Z[n]; }
                Z_[n] = hiddenLayers[layer].forward(x_);
                mask_[n] = dropout(Z_[n], pDrouput); // apply dropout mask to units
            }
            Z = Z_;
            layerInputs.add(Z.clone());
            dropoutMasks.add(mask_);
        }
        // After forward propagation through the hidden layers, training data is forward propagated in the output layer
        // of the logistic regression. Then, the deltas of each layer are going back through the network. Here, we apply
        // the cached masks to the delta so that its values are backprop in the same network for/back-ward output layer
        D = outputLayer.train(Z, T, minibatchSize, learningRate);
        // backward hidden layers
        for (int layer = nLayers - 1; layer >= 0; layer--) {
            double[][] Wprev_;
            if (layer == nLayers - 1) { Wprev_ = outputLayer.getW(); }
            else { Wprev_= hiddenLayers[layer + 1].getW(); }
            // apply mask to delta as well
            for (int n = 0; n < minibatchSize; n++) {
                int[] mask_ = dropoutMasks.get(layer)[n];
                for (int j = 0; j < D[n].length; j++) { D[n][j] *= mask_[j]; }
            }
            D = hiddenLayers[layer].backward(layerInputs.get(layer), layerInputs.get(layer + 1), D, Wprev_,
                    minibatchSize, learningRate);
        }
    }

    // this function returns the values following the Bernoulli distribution
    private int[] dropout(double[] z, double p) {
        int size = z.length;
        int[] mask = new int[size];
        for (int i = 0; i < size; i++) {
            mask[i] = RandomGenerator.binomial(1, 1 - p, rng);
            z[i] *= mask[i]; // apply mask
        }
        return mask;
    }

    // Before applying the test data to tuned model, we need to configure the weights of the network. Dropout masks cannot
    // be simply applied to the test data, because when masked, the shape of each network will be differentiated, and this
    // may returns different results because a certain unit may have a significant effect on certain features. Instead, we
    // need to smooth the weights of the network. Below, all the weights are multiplied by the probability of non-dropout.
    public void pretest(double pDropout) {
        for (int layer = 0; layer < nLayers; layer++) {
            int nIn_, nOut_;
            if (layer == 0) { nIn_ = nIn; }
            else { nIn_ = hiddenLayerSizes[layer]; }
            if (layer == nLayers - 1) { nOut_ = nOut; }
            else { nOut_ = hiddenLayerSizes[layer+1]; }
            for (int j = 0; j < nOut_; j++) {
                for (int i = 0; i < nIn_; i++) {
                    hiddenLayers[layer].getW()[j][i] *= 1 - pDropout;
                }
            }
        }
    }

    public Integer[] predict(double[] x) {
        double[] z = new double[0];
        for (int layer = 0; layer < nLayers; layer++) {
            double[] x_;
            if (layer == 0) { x_ = x; }
            else { x_ = z.clone(); }
            z = hiddenLayers[layer].forward(x_);
        }
        return outputLayer.predict(z);
    }

    /** Getters ans Setters */
    public int getnIn() { return nIn; }
    public void setnIn(int nIn) { this.nIn = nIn; }
    public int getnOut() { return nOut; }
    public void setnOut(int nOut) { this.nOut = nOut; }
    public int[] getHiddenLayerSizes() { return hiddenLayerSizes; }
    public void setHiddenLayerSizes(int[] hiddenLayerSizes) { this.hiddenLayerSizes = hiddenLayerSizes; }
    public int getnLayers() { return nLayers; }
    public void setnLayers(int nLayers) { this.nLayers = nLayers; }
    public Random getRng() { return rng; }
    public void setRng(Random rng) { this.rng = rng; }

}
