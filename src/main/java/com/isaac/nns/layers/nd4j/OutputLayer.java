package com.isaac.nns.layers.nd4j;

import com.isaac.nns.initialization.Activation;
import com.isaac.nns.initialization.ActivationFunction;
import com.isaac.nns.initialization.Weight;
import com.isaac.nns.initialization.WeightInitialization;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;
import java.util.function.Function;

/**
 * Created by zhanghao on 19/6/17.
 * @author ZHANG HAO
 */
public class OutputLayer {
    public int nIn;
    public int nOut;
    public INDArray W;
    public INDArray b;
    public Random rng;
    private Function<INDArray, INDArray> activation;

    public OutputLayer (int nIn, int nOut, Weight weight, Activation activation, Random rng) {
        this.nIn = nIn;
        this.nOut = nOut;
        this.rng = rng == null ? new Random(12345) : rng;
        this.W = WeightInitialization.initialize(nIn, nOut, this.rng, weight);
        this.b = Nd4j.create(new double[nOut], new int[] {nOut, 1});
        this.activation = ActivationFunction.activations(activation);
    }

    public INDArray train(INDArray X, INDArray T, int minibatchSize, double learningRate) {
        // train with SGD:
        // 1. Calculate the gradient of W and b using the data from the mini-batch
        INDArray dY = output(X).sub(T);
        // 2. Update W and b with the gradients
        W.subi(dY.transpose().mmul(X).mul(Nd4j.scalar(learningRate / minibatchSize)));
        b.subi(dY.sum(0).transpose().mul(Nd4j.scalar(learningRate / minibatchSize)));
        return dY;
    }

    private INDArray output(INDArray X) {
        INDArray preActivation = X.mmul(W.transpose()).addRowVector(b.transpose()); // minibatchSize * nOut
        return activation.apply(preActivation);
    }

    public INDArray predict(INDArray x) {
        INDArray y = output(x); // activate input data through learned networks
        INDArray out = Nd4j.create(new double[x.rows() * nOut], new int[] { x.rows(), nOut });
        for (int i = 0; i < x.rows(); i++) {
            int argmax = -1;
            double max = Double.MIN_VALUE;
            for (int j = 0; j < nOut; j++) {
                if (max < y.getDouble(i, j)) {
                    argmax = j;
                    max = y.getDouble(i, j);
                }
            }
            out.put(i, argmax, Nd4j.scalar(1.0));
        }
        return out;
    }
}
