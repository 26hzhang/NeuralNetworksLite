package com.isaac.java.utils;

public final class ActivationFunction {

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
