package com.isaac.java.neuralnetworks;

import java.util.Random;

import com.isaac.java.utils.ActivationFunction;
import com.isaac.nd4j.utils.RandomGenerator;

public class RestrictedBoltzmannMachine {
	public int nVisible;
    public int nHidden;
    public double[][] W;
    public double[] hbias;
    public double[] vbias;
    public Random rng;
    
    public RestrictedBoltzmannMachine(int nVisible, int nHidden, double[][] W, double[] hbias, double[] vbias, 
    		Random rng) {
   	
    	if (W == null) {
    		W = new double[nHidden][nVisible];
    		double w = 1.0 / nVisible;
    		for (int j = 0; j < nHidden; j++) {
    			for (int i = 0; i < nVisible; i++) {
    				W[j][i] = RandomGenerator.uniform(-w, w, rng);
    			}
    		}
    	}
    	
    	if (hbias == null) {
    		hbias = new double[nHidden];
    		for (int j = 0; j < nHidden; j++)
    			hbias[j] = 0.0;
    	}
    	
    	if (vbias == null) {
    		vbias = new double[nVisible];
    		for (int i = 0; i < nVisible; i++)
    			vbias[i] = 0.0;
    	}
    	
    	if (rng == null)
    		rng = new Random(1234);
    	
    	this.nVisible = nVisible;
    	this.nHidden = nHidden;
    	this.W = W;
    	this.hbias = hbias;
    	this.vbias = vbias;
    	this.rng = rng;
    }

    public void contrastiveDivergence(int[][] X, int minibatchSize, double learningRate, int k) {
    	double[][] grad_W = new double[nHidden][nVisible];
        double[] grad_hbias = new double[nHidden];
        double[] grad_vbias = new double[nVisible];
        // train with minibatches
        for (int n = 0; n < minibatchSize; n++) {
        	double[] phMean_ = new double[nHidden];
            int[] phSample_ = new int[nHidden];
            double[] nvMeans_ = new double[nVisible];
            int[] nvSamples_ = new int[nVisible];
            double[] nhMeans_ = new double[nHidden];
            int[] nhSamples_ = new int[nHidden];
            // train with contrastive divergence
            // CD-k: CD-1 is enough for sampling (i.e. k == 1)
            sampleHgivenV(X[n], phMean_, phSample_);
            for (int step = 0; step < k; step++) {
            	// Gibbs sampling
            	if (step == 0) {
            		gibbsHVH(phSample_, nvMeans_, nvSamples_, nhMeans_, nhSamples_);
            	} else {
            		gibbsHVH(nhSamples_, nvMeans_, nvSamples_, nhMeans_, nhSamples_);
            	}
            }
            // calculate gradients
            for (int j = 0; j < nHidden; j++) {
            	for (int i = 0; i < nVisible; i++) {
            		grad_W[j][i] += phMean_[j] * X[n][i] - nhMeans_[j] * nvSamples_[i];
            	}
            	grad_hbias[j] += phMean_[j] - nhMeans_[j];
            }
            for (int i = 0; i < nVisible; i++) {
            	grad_vbias[i] += X[n][i] - nvSamples_[i];
            }
        }
        // update parameters
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
    
    public void gibbsHVH(int[] h0Sample, double[] nvMeans, int[] nvSamples, double[] nhMeans, int[] nhSamples) {
    	sampleVgivenH(h0Sample, nvMeans, nvSamples);
    	sampleHgivenV(nvSamples, nhMeans, nhSamples);
    }
    
    public void sampleHgivenV(int[] v0Sample, double[] mean, int[] sample) {
    	for (int j = 0; j < nHidden; j++) {
    		mean[j] = propup(v0Sample, W[j], hbias[j]);
    		sample[j] = RandomGenerator.binomial(1, mean[j], rng);
    	}
    }
    
    public void sampleVgivenH(int[] h0Sample, double[] mean, int[] sample) {
    	for (int i = 0; i < nVisible; i++) {
    		mean[i] = propdown(h0Sample, i, vbias[i]);
    		sample[i] = RandomGenerator.binomial(1, mean[i], rng);
    	}
    }
    
    public double propup(int[] v, double[] w, double bias) {
    	double preActivation = 0.0;
    	for (int i = 0; i < nVisible; i++) {
    		preActivation += w[i] * v[i];
    	}
    	preActivation += bias;
    	return ActivationFunction.sigmoid(preActivation);
    }
    
    public double propdown(int[] h, int i, double bias) {
    	double preActivation = 0.0;
    	for (int j = 0; j < nHidden; j++) {
    		preActivation += W[j][i] * h[j];
    	}
    	preActivation += bias;
    	return ActivationFunction.sigmoid(preActivation);
    }
    
    public double[] reconstruct(int[] v) {
    	double[] x = new double[nVisible];
    	double[] h = new double[nHidden];
    	for (int j = 0; j < nHidden; j++) {
    		h[j] = propup(v, W[j], hbias[j]);
    	}
    	for (int i = 0; i < nVisible; i++) {
    		double preActivation = 0.0;
    		for (int j = 0; j < nHidden; j++) {
    			preActivation += W[j][i] * h[j];
    		}
    		preActivation += vbias[i];
    		x[i] = ActivationFunction.sigmoid(preActivation);
    	}
    	return x;
    }
}
