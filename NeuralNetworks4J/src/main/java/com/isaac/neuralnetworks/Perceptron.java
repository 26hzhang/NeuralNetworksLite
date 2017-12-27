package com.isaac.neuralnetworks;

@SuppressWarnings("unused")
public class Perceptron {
    private int nIn; // dimensions of input data
    private double[] w; // weight vector of perceptrons
    public double learningRate;

    public Perceptron(int nIn) {
        this.nIn = nIn;
        this.w = new double[nIn];
    }

    public int train(double[] x, int t, double learningRate) {
        int classified = 0;
        double c = 0;
        // check if the data is classified correctly
        for (int i = 0; i < nIn; i++) c += w[i] * x[i] * t;
        // apply steepest descent method if the data is wrongly classified
        if (c > 0) classified = 1;
        else {
            for (int i = 0; i < nIn; i++) w[i] += learningRate * x[i] * t;
        }
        return classified;
    }

    public int predict(double[] x) {
        double preActivation = 0.0;
        for (int i = 0; i < nIn; i++) preActivation += w[i] * x[i];
        return preActivation > 0 ? 1 : -1;
    }

    /** Getters and Setters */
    public int getnIn() { return nIn; }
    public void setnIn(int nIn) { this.nIn = nIn; }
    public double[] getW() { return w; }
    public void setW(double[] w) { this.w = w; }
}
