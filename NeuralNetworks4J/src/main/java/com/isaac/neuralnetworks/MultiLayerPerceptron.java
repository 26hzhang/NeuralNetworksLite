package com.isaac.neuralnetworks;

import com.isaac.initialization.Activation;
import com.isaac.layers.DenseLayer;
import com.isaac.layers.LogisticLayer;

import java.util.Random;

@SuppressWarnings("unused")
public class MultiLayerPerceptron {
    private int nIn;
    private int nHidden;
    private int nOut;
    private DenseLayer hiddenLayer;
    private LogisticLayer outputLayer;
    public Random rng;

    public  MultiLayerPerceptron(int nIn, int nHidden, int nOut, Random rng, Activation activationMethod) {
        this.nIn = nIn;
        this.nHidden = nHidden;
        this.nOut = nOut;
        this.rng = rng == null ? new Random(1234) : rng;
        // construct hidden layer with tanh as activation function
        activationMethod = activationMethod == null ? Activation.Sigmoid : activationMethod;
        hiddenLayer = new DenseLayer(nIn, nHidden, null, null, rng, activationMethod);
        // construct output layer i.e. multi-class logistic layer
        outputLayer = new LogisticLayer(nHidden, nOut);
    }

    public void train(double[][] X, int T[][], int minibatchSize, double learningRate) {
        double[][] Z = new double[minibatchSize][nIn]; // outputs of hidden layer (= inputs of output layer)
        double[][] dY;
        // forward hidden layer
        for (int n = 0; n < minibatchSize; n++) {
            Z[n] = hiddenLayer.forward(X[n]); // activate input units
        }
        // forward & backward output layer
        dY = outputLayer.train(Z, T, minibatchSize, learningRate);
        // backward hidden layer (back-propagate)
        hiddenLayer.backward(X, Z, dY, outputLayer.getW(), minibatchSize, learningRate);
    }

    public Integer[] predict(double[] x) {
        double[] z = hiddenLayer.output(x);
        return outputLayer.predict(z);
    }

    /** Getters and Setters */
    public int getnIn() { return nIn; }
    public void setnIn(int nIn) { this.nIn = nIn; }
    public int getnHidden() { return nHidden; }
    public void setnHidden(int nHidden) { this.nHidden = nHidden; }
    public int getnOut() { return nOut; }
    public void setnOut(int nOut) { this.nOut = nOut; }
    public Random getRng() { return rng; }
    public void setRng(Random rng) { this.rng = rng; }

}
