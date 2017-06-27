package com.isaac.dl4j.kaggle.houseprices;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;

/**
 * Created by zhanghao on 22/6/17.
 * @author ZHANG HAO
 *
 * Kaggle Website:
 * House Prices: Advanced Regression Techniques link: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
 * Data Pre-process (cleaning, one-hot encoding, pca, log transform) is done by Python (cause Python is much more friendly
 * in data preprocess)
 */
public class DNNRegression {

    public static final Logger log = LoggerFactory.getLogger(DNNRegression.class);

    private static final int iterations = 1;
    private static final double learningRate = 0.05;
    private static final int seed = 123;
    private static final int batchSize = 64;
    private static final int epochs = 100;

    public static void main (String[] args) throws IOException {
        log.info("load training data and labels to DataSetIterator...");
        DataSetIterator iterator = DataLoader.getTrainingData(batchSize);

        log.info("load test data...");
        File testFile = new ClassPathResource("House_Prices/test_pca.txt").getFile();
        INDArray test = DataLoader.loadData(testFile);

        final int nIn = iterator.inputColumns();
        final int nOut = 1;

        log.info("build multi-layer network...");
        MultiLayerNetwork model = constructNNModel(nIn, nOut);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        log.info("training...");
        for (int epoch = 0; epoch < epochs; epoch++) {
            iterator.reset(); // reset iterator
            model.fit(iterator);
        }

        log.info("predicting...");
        double[] predicts = new double[test.rows()];
        for (int i = 0; i < test.rows(); i++) {
            predicts[i] = Math.pow(Math.E, model.output(test.getRow(i), false).getDouble(0));
            log.info(String.valueOf(predicts[i]));
        }

        log.info("write to submission.csv...");
        File outputFile = new ClassPathResource("House_Prices/submission.csv").getFile();
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile)));
        writer.write("Id,SalePrice\n");
        int startId = 1461;
        for (int i = 0; i < predicts.length; i++) {
            writer.write((startId + i) + "," + predicts[i] + "\n");
        }
        writer.close();
        log.info("done...");
    }

    private static MultiLayerNetwork constructNNModel (int nIn, int nOut) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .momentum(0.9)
                .learningRate(learningRate)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(nIn)
                        .nOut(30)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nOut(10)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new OutputLayer.Builder()
                        .nOut(nOut)
                        .activation(Activation.IDENTITY) // for regression, activation of output layer set as IDENTITY -- f(x)=x
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .backprop(true)
                .pretrain(false)
                .build();
        return new MultiLayerNetwork(conf);
    }
}
