package com.isaac.initialization;

import com.isaac.utils.GaussianDistribution;
import com.isaac.utils.RandomGenerator;

import java.util.Random;

public enum WeightInit {
    ZERO, UNIFORM, XAVIER, XAVIER_FAN_IN, XAVIER_UNIFORM, SIGMOID_UNIFORM, RELU, RELU_UNIFORM;

    public static double[][] apply (int nIn, int nOut, WeightInit weightInit) {
        switch (weightInit) {
            case ZERO: return zero(nIn, nOut);
            case UNIFORM: return uniform(nIn, nOut);
            case XAVIER: return xavier(nIn, nOut);
            case XAVIER_FAN_IN: return xavierFanIn(nIn, nOut);
            case XAVIER_UNIFORM: return xavierUniform(nIn, nOut);
            case SIGMOID_UNIFORM: return sigmoidUniform(nIn, nOut);
            case RELU: return relu(nIn, nOut);
            case RELU_UNIFORM: return reluUniform(nIn, nOut);
            default: throw new IllegalArgumentException("Given weight initialing method not found or un-supported");
        }
    }

    private static double[][] zero (int nIn, int nOut) {
        double[][] weight = new double[nOut][nIn];
        for (int i = 0; i < nOut; i++) {
            for (int j = 0; j < nIn; j++) weight[i][j] = 0.0;
        }
        return weight;
    }

    private static double[][] uniform (int nIn, int nOut) {
        double[][] weight = new double[nOut][nIn];
        double w = Math.sqrt(1.0 / nIn);
        Random rng = new Random(12345);
        for (int i = 0; i < nOut; i++) {
            for (int j = 0; j < nIn; j++) weight[i][j] = RandomGenerator.uniform(-w, w, rng);
        }
        return weight;
    }

    private static double[][] xavier (int nIn, int nOut) {
        double[][] weight = new double[nOut][nIn];
        Random rng = new Random(12345);
        GaussianDistribution gauss = new GaussianDistribution(0.0, Math.sqrt(2.0 / (nIn + nOut)), rng);
        for (int i = 0; i < nOut; i++) {
            for (int j = 0; j < nIn; j++) weight[i][j] = gauss.random();
        }
        return weight;
    }

    private static double[][] xavierFanIn (int nIn, int nOut) {
        double[][] weight = new double[nOut][nIn];
        Random rng = new Random(12345);
        GaussianDistribution gauss = new GaussianDistribution(0.0, Math.sqrt(1.0 / nIn), rng);
        for (int i = 0; i < nOut; i++) {
            for (int j = 0; j < nIn; j++) weight[i][j] = gauss.random();
        }
        return weight;
    }

    private static double[][] xavierUniform (int nIn, int nOut) {
        double[][] weight = new double[nOut][nIn];
        Random rng = new Random(12345);
        double w = Math.sqrt(6.0 / (nIn + nOut));
        for (int i = 0; i < nOut; i++) {
            for (int j = 0; j < nIn; j++) weight[i][j] = RandomGenerator.uniform(-w, w, rng);
        }
        return weight;
    }

    private static double[][] sigmoidUniform (int nIn, int nOut) {
        double w = Math.sqrt(6.0 / (nIn + nOut)) * 4.0;
        Random rng = new Random(12345);
        double[][] weight = new double[nOut][nIn];
        for (int i = 0; i < nOut; i++) {
            for (int j = 0; j < nIn; j++) weight[i][j] = RandomGenerator.uniform(-w, w, rng);
        }
        return weight;
    }

    private static double[][] relu (int nIn, int nOut) {
        double[][] weight = new double[nOut][nIn];
        Random rng = new Random(12345);
        GaussianDistribution gauss = new GaussianDistribution(0.0, Math.sqrt(2.0 / nIn), rng);
        for (int i = 0; i < nOut; i++) {
            for (int j = 0; j < nIn; j++) weight[i][j] = gauss.random();
        }
        return weight;
    }

    private static double[][] reluUniform (int nIn, int nOut) {
        double[][] weight = new double[nOut][nIn];
        double w = Math.sqrt(6.0 / nIn);
        Random rng = new Random(12345);
        for (int i = 0; i < nOut; i++) {
            for (int j = 0; j < nIn; j++) weight[i][j] = RandomGenerator.uniform(-w, w, rng);
        }
        return weight;
    }
}
