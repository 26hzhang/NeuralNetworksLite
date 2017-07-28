package com.isaac.nns.layers.basic;

import com.isaac.nns.initialization.ActivationFunction;
import com.isaac.nns.utils.RandomGenerator;

import java.util.Random;


public class DenoisingAutoencoder {
	public int nVisible;
	public int nHidden;
	public double[][] W;
	public double[] vbias;
	public double[] hbias;
	public Random rng;
	
	public DenoisingAutoencoder(int nVisible, int nHidden, double[][] W, double[] vbias, double[] hbias, Random rng) {
		this.nVisible = nVisible;
		this.nHidden = nHidden;
		if (rng == null)
			rng = new Random(1234); // seed random
		if (W == null) {
			W = new double[nHidden][nVisible];
			double w_ = 1.0 / nVisible;
			for (int j = 0; j < nHidden; j++) {
				for (int i = 0; i < nVisible; i++) {
					W[j][i] = RandomGenerator.uniform(-w_, w_, rng);
				}
			}
		}
		if (vbias == null) {
			vbias = new double[nVisible];
		}
		if (hbias == null) {
			hbias = new double[nHidden];
		}
		this.W = W;
		this.vbias = vbias;
		this.hbias = hbias;
		this.rng = rng;
	}
	
	public void train(double[][] X, int minibatchSize, double learningRate, double corruptionLevel) {
        double[][] grad_W = new double[nHidden][nVisible];
        double[] grad_hbias = new double[nHidden];
        double[] grad_vbias = new double[nVisible];
        // train with minibatches
        for (int n = 0; n < minibatchSize; n++) {
            // add noise to original inputs
            double[] corruptedInput = getCorruptedInput(X[n], corruptionLevel);
            // encode
            double[] z = getHiddenValues(corruptedInput);
            // decode
            double[] y = getReconstructedInput(z);
            // calculate gradients
            // vbias
            double[] v_ = new double[nVisible];
            for (int i = 0; i < nVisible; i++) {
                v_[i] = X[n][i] - y[i];
                grad_vbias[i] += v_[i];
            }
            // hbias
            double[] h_ = new double[nHidden];
            for (int j = 0; j < nHidden; j++) {
                for (int i = 0; i < nVisible; i++) {
                    h_[j] = W[j][i] * (X[n][i] - y[i]);
                }
                h_[j] *= z[j] * (1 - z[j]);
                grad_hbias[j] += h_[j];
            }
            // W
            for (int j = 0; j < nHidden; j++) {
                for (int i = 0; i < nVisible; i++) {
                    grad_W[j][i] += h_[j] * corruptedInput[i] + v_[i] * z[j];
                }
            }
        }
        // update params
        for (int j = 0; j < nHidden; j++) {
            for (int i = 0; i < nVisible; i++) {
                W[j][i] += learningRate * grad_W[j][i] / minibatchSize;
            }
            hbias[j] += learningRate * grad_hbias[j] / minibatchSize;
        }
        for (int i = 0; i < nVisible; i++) {
            vbias[i] += learningRate * grad_vbias[i] / minibatchSize;
        }
    }

    public double[] getCorruptedInput(double[] x, double corruptionLevel) {
        double[] corruptedInput = new double[x.length];
        // add masking noise
        for (int i = 0; i < x.length; i++) {
            double rand_ = rng.nextDouble();
            if (rand_ < corruptionLevel) {
                corruptedInput[i] = 0.;
            } else {
                corruptedInput[i] = x[i];
            }
        }
        return corruptedInput;
    }

    public double[] getHiddenValues(double[] x) {
        double[] z = new double[nHidden];
        for (int j = 0; j < nHidden; j++) {
            for (int i = 0; i < nVisible; i++) {
                z[j] += W[j][i] * x[i];
            }
            z[j] += hbias[j];
            z[j] = ActivationFunction.sigmoid(z[j]);
        }
        return z;
    }

    public double[] getReconstructedInput(double[] z) {
        double[] y = new double[nVisible];
        for (int i = 0; i < nVisible; i++) {
            for (int j = 0; j < nHidden; j++) {
                y[i] += W[j][i] * z[j];
            }
            y[i] += vbias[i];
            y[i] = ActivationFunction.sigmoid(y[i]);
        }
        return y;
    }

    public double[] reconstruct(double[] x) {
        double[] z = getHiddenValues(x);
        double[] y = getReconstructedInput(z);
        return y;
    }
}
