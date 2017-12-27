package com.isaac.layers;

import com.isaac.initialization.Activation;
import com.isaac.initialization.BiasInit;
import com.isaac.initialization.WeightInit;

import java.util.Arrays;

@SuppressWarnings("unused")
public class LogisticLayer {
    private int nIn;
    private int nOut;
    private double[][] W;
    private double[] b;

    public LogisticLayer (int nIn, int nOut) {
        this.nIn = nIn;
        this.nOut = nOut;
        this.W = WeightInit.apply(nIn, nOut, WeightInit.UNIFORM);
        this.b = BiasInit.apply(nOut, null, BiasInit.ZERO);
    }

    public double[][] train(double[][] X, int T[][], int minibatchSize, double learningRate) {
        double[][] grad_W = new double[nOut][nIn];
        double[] grad_b = new double[nOut];
        double[][] dY = new double[minibatchSize][nOut];
        for (int n = 0; n < minibatchSize; n++) { // 1. calculate gradient of W, b
            double[] predicted_Y_ = output(X[n]);
            for (int j = 0; j < nOut; j++) {
                dY[n][j] = predicted_Y_[j] - T[n][j];
                for (int i = 0; i < nIn; i++) grad_W[j][i] += dY[n][j] * X[n][i];
                grad_b[j] += dY[n][j];
            }
        }
        for (int j = 0; j < nOut; j++) { // 2. update params
            for (int i = 0; i < nIn; i++) W[j][i] -= learningRate * grad_W[j][i] / minibatchSize;
            b[j] -= learningRate * grad_b[j] / minibatchSize;
        }
        return dY;
    }

    private double[] output(double[] x) {
        double[] preActivation = new double[nOut];
        for (int j = 0; j < nOut; j++) {
            for (int i = 0; i < nIn; i++) preActivation[j] += W[j][i] * x[i];
            preActivation[j] += b[j]; // linear output
        }
        return Activation.softmax(preActivation);
    }

    public Integer[] predict(double[] x) {
        double[] y = output(x); // activate input data through learned networks
        double max = Arrays.stream(y).max().orElse(Double.MAX_VALUE);
        Integer[] t = new Integer[nOut]; // output is the probability, so cast it to label
        for (int i = 0; i < nOut; i++) {
            if (y[i] == max) t[i] = 1;
            else t[i] = 0;
        }
        return t;
    }

    /** Getters and Setters */
    public int getnIn() { return nIn; }
    public void setnIn(int nIn) { this.nIn = nIn; }
    public int getnOut() { return nOut; }
    public void setnOut(int nOut) { this.nOut = nOut; }
    public double[][] getW() { return W; }
    public void setW(double[][] w) { this.W = w; }
    public double[] getB() { return b; }
    public void setB(double[] b) { this.b = b; }
}
