package com.isaac.layers;

import com.isaac.initialization.Activation;
import com.isaac.initialization.BiasInit;
import com.isaac.initialization.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;
import java.util.function.Function;

public class DenoisingAutoencoder {
    private int nVisible;
    private int nHidden;
    private INDArray W;
    private INDArray vbias;
    private INDArray hbias;
    private Random rng;
    private Function<INDArray, INDArray> activation; // normally, use Sigmoid

    public DenoisingAutoencoder(int nVisible, int nHidden, INDArray W, INDArray hbias, INDArray vbias, Random rng,
                                Activation activationMethod) {
        this.nVisible = nVisible;
        this.nHidden = nHidden;
        this.rng = rng == null ? new Random(12345) : rng;
        this.W = W == null ? WeightInit.apply(nVisible, nHidden, this.rng, WeightInit.UNIFORM) : W;
        this.hbias = hbias == null ? BiasInit.apply(nHidden, null, BiasInit.ZERO) : hbias;
        this.vbias = vbias == null ? BiasInit.apply(nVisible, null, BiasInit.ZERO) : vbias;
        this.activation = activationMethod == null ? Activation.active(Activation.Sigmoid) : Activation.active(activationMethod);
    }

    public void train(INDArray X, int minibatchSize, double learningRate, double corruptionLevel) {
        // 1. add noise to original inputs
        INDArray corruptedInput = getCorruptedInput(X, corruptionLevel);
        // 2. encode
        INDArray z = getHiddenValues(corruptedInput);
        // 3. decode
        INDArray y = getReconstructedInput(z);
        // calculate gradients
        // update vbias
        INDArray v_ = X.sub(y);
        vbias.addi(v_.sum(0).transpose().mul(Nd4j.scalar(learningRate / minibatchSize)));
        // update hbias
        INDArray h_ = v_.mmul(W.transpose()).mul(z).mul(z.sub(Nd4j.scalar(1)).mul(Nd4j.scalar(-1.0)));
        hbias.addi(h_.sum(0).transpose().mul(Nd4j.scalar(learningRate / minibatchSize)));
        // update W
        W.addi(h_.transpose().mmul(corruptedInput).add(z.transpose().mmul(v_)).mul(Nd4j.scalar(learningRate / minibatchSize)));
    }

    private INDArray getCorruptedInput(INDArray x, double corruptionLevel) {
        INDArray corruptedInput = Nd4j.create(new double[x.rows() * x.columns()], new int[] { x.rows(), x.columns() });
        // add masking noise
        for (int i = 0; i < x.rows(); i++) {
            for (int j = 0; j < x.columns(); j++) {
                double rand_ = rng.nextDouble();
                if (rand_ < corruptionLevel) { corruptedInput.put(i, j, Nd4j.scalar(0.0)); }
                else { corruptedInput.put(i, j, x.getDouble(i, j)); }
            }
        }
        return corruptedInput;
    }

    private INDArray getHiddenValues(INDArray x) { return activation.apply(x.mmul(W.transpose()).addRowVector(hbias.transpose())); }

    private INDArray getReconstructedInput(INDArray z) { return activation.apply(z.mmul(W).addRowVector(vbias.transpose())); }

    public INDArray reconstruct(INDArray x) { return getReconstructedInput(getHiddenValues(x)); }

    /** Getters and Setters */
    public int getnVisible() { return nVisible; }
    public void setnVisible(int nVisible) { this.nVisible = nVisible; }
    public int getnHidden() { return nHidden; }
    public void setnHidden(int nHidden) { this.nHidden = nHidden; }
    public INDArray getW() { return W; }
    public void setW(INDArray w) { W = w; }
    public INDArray getVbias() { return vbias; }
    public void setVbias(INDArray vbias) { this.vbias = vbias; }
    public INDArray getHbias() { return hbias; }
    public void setHbias(INDArray hbias) { this.hbias = hbias; }
    public Random getRng() { return rng; }
    public void setRng(Random rng) { this.rng = rng; }
}
