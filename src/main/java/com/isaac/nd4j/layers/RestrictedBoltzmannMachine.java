package com.isaac.nd4j.layers;

import com.isaac.nd4j.initialization.Activation;
import com.isaac.nd4j.initialization.ActivationFunction;
import com.isaac.nd4j.initialization.Weight;
import com.isaac.nd4j.initialization.WeightInitialization;
import com.isaac.nd4j.utils.RandomGenerator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;
import java.util.function.Function;

/**
 * Created by zhanghao on 20/6/17.
 * @author ZHANG HAO
 */
public class RestrictedBoltzmannMachine {
    public int nVisible;
    public int nHidden;
    public INDArray W;
    public INDArray hbias;
    public INDArray vbias;
    public Random rng;
    private Function<INDArray, INDArray> activation;

    private INDArray phMean_;
    private INDArray phSample_;
    private INDArray nvMeans_;
    private INDArray nvSamples_;
    private INDArray nhMeans_;
    private INDArray nhSamples_;

    public RestrictedBoltzmannMachine (int nVisible, int nHidden, Weight weight, Activation activation, Random rng) {
        this.nVisible = nVisible;
        this.nHidden = nHidden;
        this.rng = rng == null ? new Random(12345) : rng;
        this.W = WeightInitialization.initialize(nVisible, nHidden, this.rng, weight);
        this.vbias = Nd4j.create(new double[nVisible], new int[] {nVisible, 1});
        this.hbias = Nd4j.create(new double[nHidden], new int[] {nHidden, 1});
        this.activation = ActivationFunction.activations(activation);
    }

    public RestrictedBoltzmannMachine (int nVisible, int nHidden, INDArray W, INDArray hbias, INDArray vbias, Activation activation, Random rng) {
        this.nVisible = nVisible;
        this.nHidden = nHidden;
        this.W = W;
        this.hbias = hbias == null ? Nd4j.create(new double[nHidden], new int[] {nHidden, 1}) : hbias;
        this.vbias = vbias == null ? Nd4j.create(new double[nVisible], new int[] {nVisible, 1}) : vbias;
        this.rng = rng == null ? new Random(12345) : rng;
        this.activation = ActivationFunction.activations(activation);
    }

    public void contrastiveDivergence(INDArray X, int minibatchSize, double learningRate, int k) {
        this.phMean_ = Nd4j.create(new double[minibatchSize * nHidden], new int[] { minibatchSize, nHidden });
        this.phSample_ = Nd4j.create(new double[minibatchSize * nHidden], new int[] { minibatchSize, nHidden });
        this.nvMeans_ = Nd4j.create(new double[minibatchSize * nVisible], new int[] { minibatchSize, nVisible });
        this.nvSamples_ = Nd4j.create(new double[minibatchSize * nVisible], new int[] { minibatchSize, nVisible });
        this.nhMeans_ = Nd4j.create(new double[minibatchSize * nHidden], new int[] { minibatchSize, nHidden });
        this.nhSamples_ = Nd4j.create(new double[minibatchSize * nHidden], new int[] { minibatchSize, nHidden });
        // sample H given V
        this.phMean_ = propup(X);
        this.phSample_ = binomial(this.phMean_, rng);
        // train with contrastive divergence, CD-k: CD-1 is enough for sampling (i.e. k == 1)
        for (int step = 0; step < k; step++) {
            // Gibbs sampling
            if (step == 0) {
                sampleVgivenH(this.phSample_);
                sampleHgivenV(this.nvSamples_);
            } else {
                sampleVgivenH(this.nhSamples_);
                sampleHgivenV(this.nvSamples_);
            }
        }
        // update parameters
        W.addi(phMean_.transpose().mmul(X).sub(nhMeans_.transpose().mmul(nvSamples_)).mul(Nd4j.scalar(learningRate / minibatchSize)));
        hbias.addi(phMean_.sum(0).sub(nhMeans_.sum(0)).transpose().mul(Nd4j.scalar(learningRate / minibatchSize)));
        vbias.addi(X.sum(0).sub(nvSamples_.sum(0)).transpose().mul(Nd4j.scalar(learningRate / minibatchSize)));
    }

    private void sampleHgivenV(INDArray v0Sample) {
        this.nhMeans_ = propup(v0Sample);
        this.nhSamples_ = binomial(this.nhMeans_, rng);
    }

    private void sampleVgivenH(INDArray h0Sample) {
        this.nvMeans_ = propdown(h0Sample);
        this.nvSamples_ = binomial(this.nvMeans_, rng);
    }

    public INDArray reconstruct(INDArray v) {
        INDArray h = propup(v);
        return activation.apply(h.mmul(W).addRowVector(vbias.transpose()));
    }

    private INDArray propup(INDArray v) {
        return activation.apply(v.mmul(W.transpose()).addRowVector(hbias.transpose()));
    }

    private INDArray propdown(INDArray h) {
        return activation.apply(h.mmul(W).addRowVector(vbias.transpose()));
    }

    private INDArray binomial(INDArray x, Random rng) {
        INDArray y = Nd4j.create(new double[x.rows() * x.columns()], new int[] { x.rows(), x.columns() });
        for (int i = 0; i < x.rows(); i++) {
            for (int j = 0; j < x.columns(); j++) {
                y.put(i, j, RandomGenerator.binomial(1, x.getDouble(i, j), rng));
            }
        }
        return y;
    }
}
