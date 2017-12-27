package com.isaac.layers;

import com.isaac.initialization.Activation;
import com.isaac.initialization.BiasInit;
import com.isaac.initialization.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;
import java.util.function.Function;

@SuppressWarnings("unused")
public class OutputLayer {
    private int nIn;
    private int nOut;
    private INDArray W;
    private INDArray b;
    private Random rng;
    private Function<INDArray, INDArray> activation;

    public OutputLayer (int nIn, int nOut, WeightInit weight, Random rng, Activation activationMethod) {
        this.nIn = nIn;
        this.nOut = nOut;
        this.rng = rng == null ? new Random(1234) : rng;
        weight = weight == null ? WeightInit.UNIFORM : weight;
        this.W = WeightInit.apply(nIn, nOut, this.rng, weight);
        this.b = BiasInit.apply(nOut, null, BiasInit.ZERO);
        activationMethod = activationMethod == null ? Activation.Softmax : activationMethod;
        this.activation = Activation.active(activationMethod);
    }

    public INDArray train(INDArray X, INDArray T, int minibatchSize, double learningRate) {
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
