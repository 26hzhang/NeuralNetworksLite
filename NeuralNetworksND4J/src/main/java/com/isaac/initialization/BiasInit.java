package com.isaac.initialization;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public enum BiasInit {
    CONSTANT, ZERO;

    public static INDArray apply (int nOut, Double val, BiasInit biasInit) {
        val = val == null ? 0.0 : val;
        switch (biasInit) {
            case CONSTANT: return constant(nOut, val);
            case ZERO: return zero(nOut);
            default: throw new IllegalArgumentException("Given weight initialing method not found or un-supported");
        }
    }

    private static INDArray constant (int nOut, Double val) { return Nd4j.zeros(nOut, 1).addi(val); }

    private static INDArray zero (int nOut) { return Nd4j.zeros(nOut, 1); }

}
