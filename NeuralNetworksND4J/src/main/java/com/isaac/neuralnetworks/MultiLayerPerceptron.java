package com.isaac.neuralnetworks;

import com.isaac.initialization.Activation;
import com.isaac.initialization.WeightInit;
import com.isaac.layers.DenseLayer;
import com.isaac.layers.OutputLayer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Random;

public class MultiLayerPerceptron {
    private int nIn;
    private int nHidden;
    private int nOut;
    private DenseLayer hiddenLayer;
    private OutputLayer outputLayer;
    private Random rng;

    public MultiLayerPerceptron(int nIn, int nHidden, int nOut, Random rng) {
        this.nIn = nIn;
        this.nHidden = nHidden;
        this.nOut = nOut;
        this.rng = rng == null ? new Random(1234) : rng;
        this.hiddenLayer = new DenseLayer(nIn, nHidden, null, null, rng, Activation.Tanh);
        this.outputLayer = new OutputLayer(nHidden, nOut, WeightInit.UNIFORM, rng, Activation.Softmax);
    }

    public void train(INDArray X, INDArray T, int minibatchSize, double learningRate) {
        INDArray Z = hiddenLayer.forward(X);
        INDArray dY = outputLayer.train(Z, T, minibatchSize, learningRate);
        hiddenLayer.backward(X, Z, dY, outputLayer.getW(), minibatchSize, learningRate);
    }

    public INDArray predict(INDArray x) {
        return outputLayer.predict(hiddenLayer.forward(x));
    }

    /** Getters and Setters */
    public int getnIn() { return nIn; }
    public void setnIn(int nIn) { this.nIn = nIn; }
    public int getnHidden() { return nHidden; }
    public void setnHidden(int nHidden) { this.nHidden = nHidden; }
    public int getnOut() { return nOut; }
    public void setnOut(int nOut) { this.nOut = nOut; }
    public Random getRng() { return rng; }
    public void setRng(Random rng) { this.rng = rng; }

}
