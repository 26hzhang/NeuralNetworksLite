package com.isaac.nd4j.initialization;

import com.isaac.nd4j.utils.GaussianDistribution;
import com.isaac.nd4j.utils.RandomGenerator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

/**
 * Created by zhanghao on 19/6/17.
 * @author ZHANG HAO
 */
public class WeightInitialization {

    public static INDArray initialize (int nIn, int nOut, Random rng, Weight weight) {
        switch (weight) {
            case ZERO: return Zero(nIn, nOut);
            case UNIFORM: return Uniform(nIn, nOut, rng);
            case XAVIER: return Xavier(nIn, nOut, rng);
            case XAVIER_FAN_IN: return XavierFanIn(nIn, nOut, rng);
            case XAVIER_UNIFORM: return XavierUniform(nIn, nOut, rng);
            case SIGMOID_UNIFORM: return SigmoidUniform(nIn, nOut, rng);
            case RELU: return ReLU(nIn, nOut, rng);
            case RELU_UNIFORM: return ReLUUniform(nIn, nOut, rng);
            default: throw new IllegalArgumentException("weight initialization not found or un-support");
        }
    }

    private static INDArray Zero (int nIn, int nOut) {
        return Nd4j.zeros(nOut, nIn);
    }

    private static INDArray Uniform (int nIn, int nOut, Random rng) {
        double w = Math.sqrt(1.0 / nIn);
        INDArray W = Nd4j.create(new double[nOut * nIn], new int[] { nOut, nIn });
        for (int i = 0; i < nOut; i++) {
            for (int j = 0; j < nIn; j++) {
                W.put(i, j, Nd4j.scalar(RandomGenerator.uniform(-w, w, rng)));
            }
        }
        return W;
    }

    private static INDArray Xavier (int nIn, int nOut, Random rng) {
        double mean = 0;
        double variance = Math.sqrt(2.0 / (nIn + nOut));
        GaussianDistribution gauss = new GaussianDistribution(mean, variance, rng);
        INDArray W = Nd4j.create(new double[nOut * nIn], new int[] {nOut, nIn});
        for (int i = 0; i < nOut; i++) {
            for (int j = 0; j < nIn; j++) {
                W.put(i, j, Nd4j.scalar(gauss.random()));
            }
        }
        return W;
    }

    private static INDArray XavierFanIn (int nIn, int nOut, Random rng) {
        double mean = 0;
        double variance = Math.sqrt(1.0 / nIn);
        GaussianDistribution gauss = new GaussianDistribution(mean, variance, rng);
        INDArray W = Nd4j.create(new double[nOut * nIn], new int[] {nOut, nIn});
        for (int i = 0; i < nOut; i++) {
            for (int j = 0; j < nIn; j++) {
                W.put(i, j, Nd4j.scalar(gauss.random()));
            }
        }
        return W;
    }

    private static INDArray XavierUniform (int nIn, int nOut, Random rng) {
        double w = Math.sqrt(6.0 / (nIn + nOut));
        INDArray W = Nd4j.create(new double[nOut * nIn], new int[] { nOut, nIn });
        for (int i = 0; i < nOut; i++) {
            for (int j = 0; j < nIn; j++) {
                W.put(i, j, Nd4j.scalar(RandomGenerator.uniform(-w, w, rng)));
            }
        }
        return W;
    }

    private static INDArray SigmoidUniform (int nIn, int nOut, Random rng) {
        double w = Math.sqrt(6.0 / (nIn + nOut)) * 4.0;
        INDArray W = Nd4j.create(new double[nOut * nIn], new int[] { nOut, nIn });
        for (int i = 0; i < nOut; i++) {
            for (int j = 0; j < nIn; j++) {
                W.put(i, j, Nd4j.scalar(RandomGenerator.uniform(-w, w, rng)));
            }
        }
        return W;
    }

    private static INDArray ReLU (int nIn, int nOut, Random rng) {
        double mean = 0;
        double variance = Math.sqrt(2.0 / nIn); // TODO -- my understanding, it is nIn
        GaussianDistribution gauss = new GaussianDistribution(mean, variance, rng);
        INDArray W = Nd4j.create(new double[nOut * nIn], new int[] {nOut, nIn});
        for (int i = 0; i < nOut; i++) {
            for (int j = 0; j < nIn; j++) {
                W.put(i, j, Nd4j.scalar(gauss.random()));
            }
        }
        return W;
    }

    private static INDArray ReLUUniform (int nIn, int nOut, Random rng) {
        double w = Math.sqrt(6.0 / nIn);
        INDArray W = Nd4j.create(new double[nOut * nIn], new int[] { nOut, nIn });
        for (int i = 0; i < nOut; i++) {
            for (int j = 0; j < nIn; j++) {
                W.put(i, j, Nd4j.scalar(RandomGenerator.uniform(-w, w, rng)));
            }
        }
        return W;
    }
}
