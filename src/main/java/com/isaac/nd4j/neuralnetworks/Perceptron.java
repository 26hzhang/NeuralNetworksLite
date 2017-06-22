package com.isaac.nd4j.neuralnetworks;

import com.isaac.nd4j.initialization.Activation;
import com.isaac.nd4j.initialization.ActivationFunction;
import com.isaac.nd4j.initialization.Weight;
import com.isaac.nd4j.initialization.WeightInitialization;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Random;
import java.util.function.Function;

/**
 * Created by zhanghao on 20/6/17.
 * @author ZHANG HAO
 */
public class Perceptron {

	public int nIn; // dimensions of input data
	public INDArray w; // weight vector of perceptrons
	private Function<INDArray, INDArray> activation = ActivationFunction.activations(Activation.Step);

	public Perceptron(int nIn, Weight weight) {
		this.nIn = nIn;
		this.w = WeightInitialization.initialize(nIn, 1, new Random(12345), weight);
	}

	public int train(INDArray x, INDArray t, double learningRate) {
		int classified = 0;
		double c = x.mmul(w.transpose()).getDouble(0) * t.getDouble(0);
		if (c > 0) {
			classified = 1;
		} else {
			w.addi(x.transpose().mul(t).mul(learningRate).transpose());
		}
		return classified;
	}

	public INDArray predict(INDArray x) {
		return activation.apply(x.mmul(w.transpose()));
	}

}
