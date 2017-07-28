package com.isaac.nns.initialization;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.function.Function;

/**
 * Created by zhanghao on 19/6/17.
 * @author ZHANG HAO
 * Activation functions for Neural Networks
 */
public class ActivationFunction {

    public static Function<INDArray, INDArray> activations (Activation activation) {
        switch (activation) {
            case Step: return (INDArray x) -> Step(x);
            case Sigmoid: return (INDArray x) -> Sigmoid(x);
            case Tanh: return (INDArray x) -> Tanh(x);
            case ReLU: return (INDArray x) -> ReLU(x);
            case LeakyReLU: return (INDArray x) -> LeakyReLU(x);
            case SoftMax: return (INDArray x) -> Softmax(x);
            case ParametricReLU: throw new IllegalArgumentException("activation un-defined");
            default: throw new IllegalArgumentException("activation not found");
        }
    }

    public static Function<INDArray, INDArray> dactivations (Activation activation) {
        switch (activation) {
            case Sigmoid: return (INDArray y) -> dSigmoid(y);
            case Tanh: return (INDArray y) -> dTanh(y);
            case ReLU: return (INDArray y) -> dReLU(y);
            case LeakyReLU: return (INDArray y) -> dLeakyReLU(y);
            case ParametricReLU: throw new IllegalArgumentException("activation un-defined");
            case SoftMax: throw new IllegalArgumentException("activation un-defined");
            default: throw new IllegalArgumentException("activation not found");
        }
    }

    private static INDArray Step(INDArray x) {
        return Transforms.sign(x);
    }

    /**
     * Sigmoid: 1.0 / (1.0 + e^(-x));
     * @param x (INDArray)
     * @return sigmoid output (INDArray)
     */
    private static INDArray Sigmoid (INDArray x) { return Transforms.sigmoid(x); }

    /**
     * Sigmoid Derivative: sigma(x) * (1-sigma(x))
     * @param y (INDArray)
     * @return sigmoid derivative (INDArray)
     */
    private static INDArray dSigmoid (INDArray y) { return y.sub(y.mul(y)); }

    /**
     * Tanh: (e^x - e^(-x)) / (e^x + e^(-x))
     * @param x (INDArray)
     * @return tanh output (INDArray)
     */
    private static INDArray Tanh (INDArray x) { return Transforms.tanh(x); }

    /**
     * Tanh Derivative: 1 - tanh(x)^2
     * @param y (INDArray) -- tanh(x)
     * @return tanh derivative (INDArray)
     */
    private static INDArray dTanh (INDArray y) {
        return y.mul(y).sub(Nd4j.scalar(1.0)).mul(Nd4j.scalar(-1.0));
    }

    /**
     * ReLU: x(i)>0 --> x, x(i)<0 --> 0
     * @param x (INDArray)
     * @return ReLU output (INDArray)
     */
    private static INDArray ReLU (INDArray x) { return Transforms.relu(x); }

    /**
     * ReLU Derivative: y>0 --> 1, y<0 --> 0
     * @param y (INDArray)
     * @return ReLU derivative (INDArray)
     */
    private static INDArray dReLU (INDArray y) {
        return (Transforms.sign(y).add(Nd4j.scalar(1)).div(Nd4j.scalar(2)));
    }

    /**
     * Leaky ReLU: x(i)>0 --> x, x(i)<0 --> 0.01 * x(i)
     * @param x (INDArray)
     * @return Leaky ReLU output (INDArray)
     */
    private static INDArray LeakyReLU (INDArray x) {
        return Transforms.leakyRelu(x);
    }

    /**
     * Leaky ReLU Derivative: y>0 --> 1, y<0 --> 0.01
     * @param y (INDArray)
     * @return Leaky ReLU derivative (INDArray)
     */
    private static INDArray dLeakyReLU (INDArray y) {
        INDArray x = Nd4j.create(new double[y.rows() * y.columns()], new int[] { y.rows(), y.columns() });
        for (int i = 0; i < y.rows(); i++) {
            for (int j = 0; j < y.columns(); j++) {
                if (y.getDouble(i, j) > 0) x.put(i, j, Nd4j.scalar(1.0));
                else x.put(i, j, Nd4j.scalar(0.01));
            }
        }
        return x;
    }

    private static INDArray Softmax (INDArray x) {
        return Nd4j.getExecutioner().execAndReturn(new SoftMax(x));
    }

    // Pure Java Activations

    public static int step(double x) {
        if (x >= 0) {
            return 1;
        } else {
            return -1;
        }
    }

    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.pow(Math.E, -x));
    }

    public static double dsigmoid(double y) {
        return y * (1.0 - y);
    }

    public static double tanh(double x) {
        return Math.tanh(x);
    }

    public static double dtanh(double y) {
        return 1.0 - y * y;
    }

    public static double ReLU(double x) {
        if(x > 0)
            return x;
        else
            return 0.0;
    }

    public static double dReLU(double y) {
        if(y > 0)
            return 1.0;
        else
            return 0.0;
    }

    public static double LeakyReLU (double x) {
        if (x > 0)
            return x;
        else
            return 0.01 * x;
    }

    public static double dLeakyReLU (double y) {
        if (y > 0)
            return 1.0;
        else
            return 0.01;
    }

    public static double[] softmax(double[] x, int n) {
        double[] y = new double[n];
        double max = 0.0;
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            if (max < x[i]) {
                max = x[i]; // prevent overflow
            }
        }
        for (int i = 0; i < n; i++) {
            y[i] = Math.exp(x[i] - max);
            sum += y[i];
        }
        for (int i = 0; i < n; i++) {
            y[i] /= sum;
        }
        return y;
    }

}
