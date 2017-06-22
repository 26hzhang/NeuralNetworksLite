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
 * Created by zhanghao on 19/6/17.
 * @author ZHANG HAO
 */
public class DenseLayer {
    public int nIn;
    public int nOut;
    public INDArray W;
    public INDArray b;
    public Random rng;
    private Function<INDArray, INDArray> activation;
    private Function<INDArray, INDArray> dactivation;

    public DenseLayer(int nIn, int nOut, Weight weight, Activation activation, Random rng) {
        this.nIn = nIn;
        this.nOut = nOut;
        this.rng = rng == null ? new Random(12345) : rng;
        this.W = WeightInitialization.initialize(nIn, nOut, this.rng, weight);
        this.b = Nd4j.create(new double[nOut], new int[] {nOut, 1}); // initialize bias with zero
        this.activation = ActivationFunction.activations(activation);
        this.dactivation = ActivationFunction.dactivations(activation);
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
    public INDArray output (INDArray X) {
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
}
