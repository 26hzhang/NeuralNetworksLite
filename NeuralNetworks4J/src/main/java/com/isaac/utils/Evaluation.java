package com.isaac.utils;

import java.util.Arrays;

public class Evaluation {
    private Integer[][] logits;
    private Integer[][] labels;
    private int examples;
    private int patterns;
    private int[][] confusionMatrix;
    private double accuracy;
    private double[] precision;
    private double[] recall;

    public Evaluation(Integer[][] logits, Integer[][] labels) {
        this.logits = logits;
        this.labels = labels;
        this.examples = labels.length;
        this.patterns = labels[0].length;
        this.confusionMatrix = new int[patterns][patterns];
        this.precision = new double[patterns];
        this.recall = new double[patterns];
        this.accuracy = 0.0;
    }

    public Evaluation fit() {
        for (int i = 0; i < examples; i++) {
            int predicted_ = Arrays.asList(logits[i]).indexOf(1);
            int actual_ = Arrays.asList(labels[i]).indexOf(1);
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
    public double[] getRecall() { return  recall; }
    public int[][] getConfusionMatrix() { return confusionMatrix; }
}
