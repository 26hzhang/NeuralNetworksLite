package com.isaac.initialization;

import java.util.function.DoubleFunction;

@SuppressWarnings("Duplicates")
public enum Activation {
    ReLU, Sigmoid, Tanh, LeakyReLU, ParametricReLU;

    public static DoubleFunction<Double> active (Activation activation) {
        switch (activation) {
            case Sigmoid: return Activation::sigmoid;
            case Tanh: return Activation::tanh;
            case ReLU:return Activation::relu;
            case ParametricReLU:
            case LeakyReLU: return Activation::leakyrelu;
            default: throw new IllegalArgumentException("Given activation method not found or un-supported");
        }
    }

    public static DoubleFunction<Double> dactive(Activation activation) {
        switch (activation) {
            case Sigmoid: return Activation::dsigmoid;
            case Tanh: return Activation::dtanh;
            case ReLU: return Activation::drelu;
            case ParametricReLU:
            case LeakyReLU: return Activation::dleakyrelu;
            default: throw new IllegalArgumentException("Given activation method not found or un-supported");
        }
    }

    public static double sigmoid(double x) { return 1.0 / (1.0 + Math.pow(Math.E, -x)); }

    public static double dsigmoid(double y) { return y * (1.0 - y); }

    public static double tanh(double x) { return Math.tanh(x); }

    public static double dtanh(double y) { return 1.0 - y * y; }

    public static double relu(double x) { return x > 0 ? x : 0.0; }

    public static double drelu(double y) { return y > 0 ? 1.0 : 0.0; }

    public static double leakyrelu(double x) { return x > 0 ? x : 0.01 * x; }

    public static double dleakyrelu(double y) { return y > 0 ? 1.0 : 0.01; }

    public static double[] softmax(double[] x) {
        int size = x.length;
        double[] y = new double[size];
        double max = 0.0;
        double sum = 0.0;
        for (double val : x) {
            if (max < val) max = val; // prevent overflow
        }
        for (int i = 0; i < size; i++) {
            y[i] = Math.exp(x[i] - max);
            sum += y[i];
        }
        for (int i = 0; i < size; i++) y[i] /= sum;
        return y;
    }

}
