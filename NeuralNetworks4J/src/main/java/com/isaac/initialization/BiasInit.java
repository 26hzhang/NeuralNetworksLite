package com.isaac.initialization;

public enum BiasInit {
    CONSTANT, ZERO;

    public static double[] apply (int nOut, Double val, BiasInit biasInit) {
        val = val == null ? 0.0 : val;
        switch (biasInit) {
            case CONSTANT: return constant(nOut, val);
            case ZERO: return zero(nOut);
            default: throw new IllegalArgumentException("Given weight initialing method not found or un-supported");
        }
    }

    private static double[] constant (int nOut, Double val) {
        double[] bias = new double[nOut];
        for (int i = 0; i < nOut; i++) bias[i] = val;
        return bias;
    }

    private static double[] zero (int nOut) {
        return constant(nOut, 0.0);
    }
}
