package com.isaac.initialization;

import com.isaac.utils.GaussianDistribution;
import com.isaac.utils.RandomGenerator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

public enum WeightInit {
    ZERO, UNIFORM, XAVIER, XAVIER_FAN_IN, XAVIER_UNIFORM, SIGMOID_UNIFORM, RELU, RELU_UNIFORM;

    public static INDArray apply (int nIn, int nOut, Random rng, WeightInit weight) {
        switch (weight) {
            case ZERO: return zero(nIn, nOut);
            case UNIFORM: return uniform(nIn, nOut, rng);
            case XAVIER: return xavier(nIn, nOut, rng);
            case XAVIER_FAN_IN: return xavierFanIn(nIn, nOut, rng);
            case XAVIER_UNIFORM: return xavierUniform(nIn, nOut, rng);
            case SIGMOID_UNIFORM: return sigmoidUniform(nIn, nOut, rng);
            case RELU: return relu(nIn, nOut, rng);
            case RELU_UNIFORM: return reluUniform(nIn, nOut, rng);
            default: throw new IllegalArgumentException("weight initialization not found or un-support");
        }
    }

    private static INDArray zero(int nIn, int nOut) {
        return Nd4j.zeros(nOut, nIn);
    }

    private static INDArray uniform(int nIn, int nOut, Random rng) {
        double w = Math.sqrt(1.0 / nIn);
        INDArray W = Nd4j.create(new double[nOut * nIn], new int[] { nOut, nIn });
        for (int i = 0; i < nOut; i++) {
            for (int j = 0; j < nIn; j++) {
                W.put(i, j, Nd4j.scalar(RandomGenerator.uniform(-w, w, rng)));
            }
        }
        return W;
    }

    private static INDArray xavier(int nIn, int nOut, Random rng) {
        double mean = 0;
        double variance = Math.sqrt(2.0 / (nIn + nOut));
        GaussianDistribution gauss = new GaussianDistribution(mean, variance, rng);
        INDArray W = Nd4j.create(new double[nOut * nIn], new int[] {nOut, nIn});
        for (int i = 0; i < nOut; i++) {
            for (int j = 0; j < nIn; j++) { W.put(i, j, Nd4j.scalar(gauss.random())); }
        }
        return W;
    }

    private static INDArray xavierFanIn(int nIn, int nOut, Random rng) {
        double mean = 0;
        double variance = Math.sqrt(1.0 / nIn);
        GaussianDistribution gauss = new GaussianDistribution(mean, variance, rng);
        INDArray W = Nd4j.create(new double[nOut * nIn], new int[] {nOut, nIn});
        for (int i = 0; i < nOut; i++) {
            for (int j = 0; j < nIn; j++) { W.put(i, j, Nd4j.scalar(gauss.random())); }
        }
        return W;
    }

    private static INDArray xavierUniform(int nIn, int nOut, Random rng) {
        double w = Math.sqrt(6.0 / (nIn + nOut));
        INDArray W = Nd4j.create(new double[nOut * nIn], new int[] { nOut, nIn });
        for (int i = 0; i < nOut; i++) {
            for (int j = 0; j < nIn; j++) { W.put(i, j, Nd4j.scalar(RandomGenerator.uniform(-w, w, rng))); }
        }
        return W;
    }

    private static INDArray sigmoidUniform(int nIn, int nOut, Random rng) {
        double w = Math.sqrt(6.0 / (nIn + nOut)) * 4.0;
        INDArray W = Nd4j.create(new double[nOut * nIn], new int[] { nOut, nIn });
        for (int i = 0; i < nOut; i++) {
            for (int j = 0; j < nIn; j++) { W.put(i, j, Nd4j.scalar(RandomGenerator.uniform(-w, w, rng))); }
        }
        return W;
    }

    private static INDArray relu(int nIn, int nOut, Random rng) {
        double mean = 0;
        double variance = Math.sqrt(2.0 / nIn);
        GaussianDistribution gauss = new GaussianDistribution(mean, variance, rng);
        INDArray W = Nd4j.create(new double[nOut * nIn], new int[] {nOut, nIn});
        for (int i = 0; i < nOut; i++) {
            for (int j = 0; j < nIn; j++) { W.put(i, j, Nd4j.scalar(gauss.random())); }
        }
        return W;
    }

    private static INDArray reluUniform(int nIn, int nOut, Random rng) {
        double w = Math.sqrt(6.0 / nIn);
        INDArray W = Nd4j.create(new double[nOut * nIn], new int[] { nOut, nIn });
        for (int i = 0; i < nOut; i++) {
            for (int j = 0; j < nIn; j++) { W.put(i, j, Nd4j.scalar(RandomGenerator.uniform(-w, w, rng))); }
        }
        return W;
    }
}
