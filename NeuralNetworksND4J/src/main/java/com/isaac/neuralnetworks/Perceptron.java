package com.isaac.neuralnetworks;

import com.isaac.initialization.Activation;
import com.isaac.initialization.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Random;
import java.util.function.Function;

@SuppressWarnings("unused")
public class Perceptron {

	private int nIn; // dimensions of input data
	private INDArray w; // weight vector of perceptrons
	private Function<INDArray, INDArray> activation;

	public Perceptron(int nIn, WeightInit weight) {
		this.nIn = nIn;
		weight = weight == null ? WeightInit.UNIFORM : weight;
		this.w = WeightInit.apply(nIn, 1, new Random(1234), weight);
		this.activation = Activation.active(Activation.Step);
	}

	public int train(INDArray x, INDArray t, double learningRate) {
		int classified = 0;
		double c = x.mmul(w.transpose()).getDouble(0) * t.getDouble(0);
		if (c > 0) { classified = 1; }
		else { w.addi(x.transpose().mul(t).mul(learningRate).transpose()); }
		return classified;
	}

	public INDArray predict(INDArray x) {
		return activation.apply(x.mmul(w.transpose()));
	}

	/** Getters and Setters */
	public int getnIn() { return nIn; }
	public void setnIn(int nIn) { this.nIn = nIn; }
	public INDArray getW() { return w; }
	public void setW(INDArray w) { this.w = w; }

}
