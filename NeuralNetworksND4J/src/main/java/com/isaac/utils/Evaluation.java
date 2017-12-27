package com.isaac.utils;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Evaluation {
    private INDArray logits;
    private INDArray labels;
    private int examples;
    private int patterns;
    private int[][] confusionMatrix;
    private double accuracy;
    private double[] precision;
    private double[] recall;

    public Evaluation (INDArray logits, INDArray labels) {
        this.logits = logits;
        this.labels = labels;
        this.examples = labels.rows();
        this.patterns = labels.columns();
        this.confusionMatrix = new int[patterns][patterns];
        this.precision = new double[patterns];
        this.recall = new double[patterns];
        this.accuracy = 0.0;
    }

    public Evaluation fit() {
        for (int i = 0; i < examples; i++) {
            int predicted_ = 0;
            int actual_ = 0;
            for (int j = 0; j < patterns; j++) {
                if (logits.getDouble(i, j) == 1.0) predicted_ = j;
            }
            for (int j = 0; j < patterns; j++) {
                if (labels.getDouble(i, j) == 1.0) actual_ = j;
            }
            confusionMatrix[actual_][predicted_] += 1;
        }
        for (int i = 0; i < patterns; i++) {
            double col_ = 0.;
            double row_ = 0.;
            for (int j = 0; j < patterns; j++) {
                if (i == j) {
                    accuracy += confusionMatrix[i][j];
                    precision[i] += confusionMatrix[j][i];
                    recall[i] += confusionMatrix[i][j];
                }
                col_ += confusionMatrix[j][i];
                row_ += confusionMatrix[i][j];
            }
            precision[i] /= col_;
            recall[i] /= row_;
        }
        accuracy /= examples;
        return this;
    }

    public double getAccuracy() { return accuracy; }
    public double[] getPrecision() { return precision; }
    public double[] getRecall() { return recall; }
}
