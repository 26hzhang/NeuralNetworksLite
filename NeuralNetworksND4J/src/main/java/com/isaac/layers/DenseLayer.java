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
public class DenseLayer {
    private int nIn;
    private int nOut;
    private INDArray W;
    private INDArray b;
    private Random rng;
    private Function<INDArray, INDArray> activation;
    private Function<INDArray, INDArray> dactivation;

    public DenseLayer(int nIn, int nOut, INDArray W, INDArray b, Random rng, Activation activationMethod) {
        this.nIn = nIn;
        this.nOut = nOut;
        this.rng = rng == null ? new Random(1234) : rng;
        this.W = W == null ? WeightInit.apply(nIn, nOut, rng, WeightInit.UNIFORM) : W;
        this.b = b == null ? BiasInit.apply(nOut, null, BiasInit.ZERO) : b;
        activationMethod = activationMethod == null ? Activation.Sigmoid : activationMethod;
        this.activation = Activation.active(activationMethod);
        this.dactivation = Activation.dactive(activationMethod);
    }

    // forward
    public INDArray forward (INDArray X) {
        return output(X);
    }

    // backward
    public INDArray backward (INDArray X, INDArray Z, INDArray dY, INDArray Wprev, int minibatchSize, double learningRate) {
        INDArray dZ = dY.mmul(Wprev).mul(dactivation.apply(Z));
        W.subi(dZ.transpose().mmul(X).mul(Nd4j.scalar(learningRate / minibatchSize)));
        b.subi(dZ.sum(0).transpose().mul(Nd4j.scalar(learningRate / minibatchSize)));
        return dZ;
    }

    // output
    private INDArray output(INDArray X) {
        return activation.apply(X.mmul(W.transpose()).addRowVector(b.transpose()));
    }

    // output-binomial
    public INDArray outputBinomial (INDArray X) {
        INDArray out = output(X);
        INDArray y = Nd4j.create(new double[out.rows() * out.columns()], new int[] { out.rows(), out.columns() });
        for (int i = 0; i < out.rows(); i++) {
            for (int j = 0; j < out.columns(); j++) {
                double value = RandomGenerator.binomial(1, out.getDouble(i, j), rng);
                y.put(i, j, Nd4j.scalar(value));
            }
        }
        return y;
    }

    /** Getters and Setters */
    public int getnIn() { return nIn; }
    public void setnIn(int nIn) { this.nIn = nIn; }
    public int getnOut() { return nOut; }
    public void setnOut(int nOut) { this.nOut = nOut; }
    public INDArray getW() { return W; }
    public void setW(INDArray w) { W = w; }
    public INDArray getB() { return b; }
    public void setB(INDArray b) { this.b = b; }
    public Random getRng() { return rng; }
    public void setRng(Random rng) { this.rng = rng; }
}
