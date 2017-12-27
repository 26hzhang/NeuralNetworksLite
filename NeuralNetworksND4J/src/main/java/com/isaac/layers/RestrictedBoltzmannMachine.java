package com.isaac.layers;

import com.isaac.initialization.Activation;
import com.isaac.initialization.BiasInit;
import com.isaac.initialization.WeightInit;
import com.isaac.utils.RandomGenerator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;
import java.util.function.Function;

@SuppressWarnings("unused")
public class RestrictedBoltzmannMachine {
    private int nVisible;
    private int nHidden;
    private INDArray W;
    private INDArray hbias;
    private INDArray vbias;
    public Random rng;
    private Function<INDArray, INDArray> activation;

    private INDArray phMean_;
    private INDArray phSample_;
    private INDArray nvMeans_;
    private INDArray nvSamples_;
    private INDArray nhMeans_;
    private INDArray nhSamples_;

    public RestrictedBoltzmannMachine (int nVisible, int nHidden, INDArray W, INDArray hbias, INDArray vbias, Random rng,
                                       Activation activationMethod) {
        this.nVisible = nVisible;
        this.nHidden = nHidden;
        this.rng = rng == null ? new Random(1234) : rng;
        this.W = W == null ? WeightInit.apply(nVisible, nHidden, this.rng, WeightInit.UNIFORM) : W;
        this.hbias = hbias == null ? BiasInit.apply(nHidden, null, BiasInit.ZERO) : hbias;
        this.vbias = vbias == null ? BiasInit.apply(nVisible, null, BiasInit.ZERO) : vbias;
        this.activation = activationMethod == null ? Activation.active(Activation.Sigmoid) : Activation.active(activationMethod);
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
        W.addi(phMean_.transpose().mmul(X).sub(nhMeans_.transpose().mmul(nvSamples_))
                .mul(Nd4j.scalar(learningRate / minibatchSize)));
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
            for (int j = 0; j < x.columns(); j++) { y.put(i, j, RandomGenerator.binomial(1, x.getDouble(i, j), rng)); }
        }
        return y;
    }

    /** Getters and Setters */
    public int getnVisible() { return nVisible; }
    public void setnVisible(int nVisible) { this.nVisible = nVisible; }
    public int getnHidden() { return nHidden; }
    public void setnHidden(int nHidden) { this.nHidden = nHidden; }
    public INDArray getW() { return W; }
    public void setW(INDArray w) { W = w; }
    public INDArray getHbias() { return hbias; }
    public void setHbias(INDArray hbias) { this.hbias = hbias; }
    public INDArray getVbias() { return vbias; }
    public void setVbias(INDArray vbias) { this.vbias = vbias; }
    public Random getRng() { return rng; }
    public void setRng(Random rng) { this.rng = rng; }
}
