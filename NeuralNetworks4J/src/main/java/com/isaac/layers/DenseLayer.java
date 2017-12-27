package com.isaac.layers;

import com.isaac.initialization.Activation;
import com.isaac.initialization.BiasInit;
import com.isaac.initialization.WeightInit;
import com.isaac.utils.RandomGenerator;

import java.util.Random;
import java.util.function.DoubleFunction;

@SuppressWarnings("unused")
public class DenseLayer {
    private int nIn;
    private int nOut;
    private double[][] W;
    private double[] b;
    private Random rng;
    private DoubleFunction<Double> activation;
    private DoubleFunction<Double> dactivation;

    public DenseLayer(int nIn, int nOut, double[][] W, double[] b, Random rng, Activation activationMethod) {
        this.rng = rng == null ? new Random(1234) : rng;
        this.nIn = nIn;
        this.nOut = nOut;
        this.W = W == null ? WeightInit.apply(nIn, nOut, WeightInit.UNIFORM) : W;
        this.b = b == null ? BiasInit.apply(nOut, null, BiasInit.ZERO) : b;
        this.activation = Activation.active(activationMethod);
        this.dactivation = Activation.dactive(activationMethod);
    }

    public double[] output(double[] x) {
        double[] y = new double[nOut];
        for (int j = 0; j < nOut; j++) {
            double preActivation_ = 0.;
            for (int i = 0; i < nIn; i++) preActivation_ += W[j][i] * x[i];
            preActivation_ += b[j];
            y[j] = activation.apply(preActivation_);
        }
        return y;
    }

    public int[] outputBinomial(int[] x, Random rng) {
        int[] y = new int[nOut];
        double[] xCast = new double[x.length];
        for (int i = 0; i < xCast.length; i++) xCast[i] = (double) x[i];
        double[] out = output(xCast);
        for (int j = 0; j < nOut; j++) y[j] = RandomGenerator.binomial(1, out[j], rng);
        return y;
    }

    public double[] forward(double[] x) { return output(x); }

    public double[][] backward(double[][] X, double[][] Z, double[][] dY, double[][] Wprev, int minibatchSize, double learningRate) {
        double[][] dZ = new double[minibatchSize][nOut]; // backpropagation error
        double[][] grad_W = new double[nOut][nIn];
        double[] grad_b = new double[nOut];
        // train with SGD
        // calculate backpropagation error to get gradient of W, b
        for (int n = 0; n < minibatchSize; n++) {
            for (int j = 0; j < nOut; j++) {
                // k < ( nOut of previous layer )
                for (int k = 0; k < dY[0].length; k++) dZ[n][j] += Wprev[k][j] * dY[n][k];
                dZ[n][j] *= dactivation.apply(Z[n][j]);
                for (int i = 0; i < nIn; i++) grad_W[j][i] += dZ[n][j] * X[n][i];
                grad_b[j] += dZ[n][j];
            }
        }
        // update params
        for (int j = 0; j < nOut; j++) {
            for (int i = 0; i < nIn; i++) W[j][i] -= learningRate * grad_W[j][i] / minibatchSize;
            b[j] -= learningRate * grad_b[j] / minibatchSize;
        }
        return dZ;
    }

    /** Getters and Setters */
    public int getnIn() { return nIn; }
    public void setnIn(int nIn) { this.nIn = nIn; }
    public int getnOut() { return nOut; }
    public void setnOut(int nOut) { this.nOut = nOut; }
    public double[][] getW() { return W; }
    public void setW(double[][] w) { W = w; }
    public double[] getB() { return b; }
    public void setB(double[] b) { this.b = b; }
    public Random getRng() { return rng; }
    public void setRng(Random rng) { this.rng = rng; }
    public DoubleFunction<Double> getActivation() { return activation; }
    public void setActivation(DoubleFunction<Double> activation) { this.activation = activation; }
    public DoubleFunction<Double> getDactivation() { return dactivation; }
    public void setDactivation(DoubleFunction<Double> dactivation) { this.dactivation = dactivation; }
}
