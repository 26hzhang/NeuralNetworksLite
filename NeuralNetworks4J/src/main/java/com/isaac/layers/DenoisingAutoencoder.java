package com.isaac.layers;

import com.isaac.initialization.Activation;
import com.isaac.initialization.BiasInit;
import com.isaac.initialization.WeightInit;

import java.util.Random;
import java.util.function.DoubleFunction;

@SuppressWarnings({"Duplicates", "unused"})
public class DenoisingAutoencoder {
    private int nVisible;
    private int nHidden;
    private double[][] W;
    private double[] vbias;
    private double[] hbias;
    private Random rng;
    private DoubleFunction<Double> activation;

    public DenoisingAutoencoder(int nVisible, int nHidden, double[][] W, double[] hbias, double[] vbias, Random rng,
                                Activation activationMethod) {
        this.nVisible = nVisible;
        this.nHidden = nHidden;
        this.W = W == null ? WeightInit.apply(nVisible, nHidden, WeightInit.UNIFORM) : W;
        this.vbias = vbias == null ? BiasInit.apply(nVisible, null, BiasInit.ZERO) : vbias;
        this.hbias = hbias == null ? BiasInit.apply(nHidden, null, BiasInit.ZERO) : hbias;
        this.rng = rng == null ? new Random(1234) : rng;
        activationMethod = activationMethod == null ? Activation.Sigmoid : activationMethod;
        this.activation = Activation.active(activationMethod);
    }

    public void train(double[][] X, int minibatchSize, double learningRate, double corruptionLevel) {
        double[][] grad_W = new double[nHidden][nVisible];
        double[] grad_hbias = new double[nHidden];
        double[] grad_vbias = new double[nVisible];
        // train with minibatches
        for (int n = 0; n < minibatchSize; n++) {
            // add noise to original inputs
            double[] corruptedInput = getCorruptedInput(X[n], corruptionLevel);
            double[] z = getHiddenValues(corruptedInput); // encode
            double[] y = getReconstructedInput(z); // decode
            // calculate gradients: vbias
            double[] v_ = new double[nVisible];
            for (int i = 0; i < nVisible; i++) {
                v_[i] = X[n][i] - y[i];
                grad_vbias[i] += v_[i];
            }
            // calculate gradients: hbias
            double[] h_ = new double[nHidden];
            for (int j = 0; j < nHidden; j++) {
                for (int i = 0; i < nVisible; i++) h_[j] = W[j][i] * (X[n][i] - y[i]);
                h_[j] *= z[j] * (1 - z[j]);
                grad_hbias[j] += h_[j];
            }
            // calculate gradients: W
            for (int j = 0; j < nHidden; j++) {
                for (int i = 0; i < nVisible; i++) grad_W[j][i] += h_[j] * corruptedInput[i] + v_[i] * z[j];
            }
        }
        // update params
        for (int j = 0; j < nHidden; j++) {
            for (int i = 0; i < nVisible; i++) W[j][i] += learningRate * grad_W[j][i] / minibatchSize;
            hbias[j] += learningRate * grad_hbias[j] / minibatchSize;
        }
        for (int i = 0; i < nVisible; i++) {
            vbias[i] += learningRate * grad_vbias[i] / minibatchSize;
        }
    }

    private double[] getCorruptedInput(double[] x, double corruptionLevel) {
        double[] corruptedInput = new double[x.length];
        // add masking noise
        for (int i = 0; i < x.length; i++) {
            double rand_ = rng.nextDouble();
            if (rand_ < corruptionLevel) corruptedInput[i] = 0.0;
            else corruptedInput[i] = x[i];
        }
        return corruptedInput;
    }

    private double[] getHiddenValues(double[] x) {
        double[] z = new double[nHidden];
        for (int j = 0; j < nHidden; j++) {
            for (int i = 0; i < nVisible; i++) z[j] += W[j][i] * x[i];
            z[j] += hbias[j];
            z[j] = activation.apply(z[j]);
        }
        return z;
    }

    private double[] getReconstructedInput(double[] z) {
        double[] y = new double[nVisible];
        for (int i = 0; i < nVisible; i++) {
            for (int j = 0; j < nHidden; j++) y[i] += W[j][i] * z[j];
            y[i] += vbias[i];
            y[i] = activation.apply(y[i]);
        }
        return y;
    }

    public double[] reconstruct(double[] x) {
        double[] z = getHiddenValues(x);
        return getReconstructedInput(z);
    }

    /** Getters and Setters */
    public int getnVisible() { return nVisible; }
    public void setnVisible(int nVisible) { this.nVisible = nVisible; }
    public int getnHidden() { return nHidden; }
    public void setnHidden(int nHidden) { this.nHidden = nHidden; }
    public double[][] getW() { return W; }
    public void setW(double[][] w) { W = w; }
    public double[] getVbias() { return vbias; }
    public void setVbias(double[] vbias) { this.vbias = vbias; }
    public double[] getHbias() { return hbias; }
    public void setHbias(double[] hbias) { this.hbias = hbias; }
    public Random getRng() { return rng; }
    public void setRng(Random rng) { this.rng = rng; }
}
