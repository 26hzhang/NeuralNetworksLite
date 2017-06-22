package com.isaac.nd4j.neuralnetworks;

import com.isaac.nd4j.initialization.Activation;
import com.isaac.nd4j.initialization.Weight;
import com.isaac.nd4j.layers.DenseLayer;
import com.isaac.nd4j.layers.OutputLayer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Random;

/**
 * Created by zhanghao on 20/6/17.
 * @author ZHANG HAO
 */
public class MultiLayerPerceptron {
    public int nIn;
    public int nHidden;
    public int nOut;
    public DenseLayer hiddenLayer;
    public OutputLayer outputLayer;
    public Random rng;

    public MultiLayerPerceptron(int nIn, int nHidden, int nOut, Random rng) {
        this.nIn = nIn;
        this.nHidden = nHidden;
        this.nOut = nOut;
        this.rng = rng == null ? new Random(12345) : rng;
        this.hiddenLayer = new DenseLayer(nIn, nHidden, Weight.UNIFORM, Activation.Tanh, rng);
        this.outputLayer = new OutputLayer(nHidden, nOut, Weight.ZERO, Activation.SoftMax, rng);
    }

    public void train(INDArray X, INDArray T, int minibatchSize, double learningRate) {
        INDArray Z = hiddenLayer.forward(X);
        INDArray dY = outputLayer.train(Z, T, minibatchSize, learningRate);
        hiddenLayer.backward(X, Z, dY, outputLayer.W, minibatchSize, learningRate);
    }

    public INDArray predict(INDArray x) {
        return outputLayer.predict(hiddenLayer.output(x));
    }
}
