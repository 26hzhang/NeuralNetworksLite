package com.isaac.dl4j;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
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
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Created by zhanghao on 10/7/17.
 * @author ZHANG HAo
 */
public class NNRegressionExample {

    private static Logger log = LoggerFactory.getLogger(NNRegressionExample.class);

    private static final int iterations = 1;
    private static final double learningRate = 0.05;
    private static final int seed = 123;
    private static final int batchSize = 64;
    private static final int epochs = 100;

    public static void main (String[] args) {
        //org.apache.log4j.BasicConfigurator.configure();
        log.info("load training and testing datasets from file...");
        INDArray train = loadData(new File("src/main/resources/nn_regress/train_pca.txt"));
        INDArray test = loadData(new File("src/main/resources/nn_regress/test_pca.txt"));
        INDArray labels = loadData(new File("src/main/resources/nn_regress/label.txt"));

        log.info("create dataset iterator...");
        DataSetIterator iterator = createDataSetIterator(train, labels, batchSize);

        final int nIn = iterator.inputColumns();
        final int nOut = 1;

        log.info("build multi-layer network...");
        MultiLayerNetwork model = constructNNModel(nIn, nOut, seed, iterations, learningRate);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        log.info("training...");
        for (int epoch = 0; epoch < epochs; epoch++) {
            iterator.reset();
            while (iterator.hasNext()) {
                DataSet dataSet = iterator.next();
                model.fit(dataSet);
            }
        }

        log.info("predicting...");
        double[] predicts = new double[test.rows()];
        for (int i = 0; i < test.rows(); i++) {
            predicts[i] = Math.pow(Math.E, model.output(test.getRow(i), false).getDouble(0));
            log.info(String.format("Id: %d, Predict: %.5f", i + 1461, predicts[i]));
        }

        log.info("done");
    }

    /**
     * Construct the Neural Networks
     * @param nIn number of input
     * @param nOut number of output
     * @param seed random seed
     * @param iterations iterations that performs
     * @param learningRate learning rate
     * @return the built network
     */
    private static MultiLayerNetwork constructNNModel (int nIn, int nOut, int seed, int iterations, double learningRate) {
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

    /**
     * Compute log, use ReLU to fix those error result (-inf)
     * @param ndArray INDArray
     * @param base base of log
     * @return new INDArray
     */
    @SuppressWarnings("unused")
    public static INDArray computeLog (INDArray ndArray, double base) {
        INDArray output = Transforms.log(ndArray, base);
        output = Transforms.relu(output); // ReLU: max(0, x) --> set -inf to zero
        return output;
    }

    /**
     * Create data set iterator for training
     * @param train training data
     * @param label labels
     * @param batchSize batch size
     * @return data set iterator
     */
    private static DataSetIterator createDataSetIterator (INDArray train, INDArray label, int batchSize) {
        DataSet dataSet = new DataSet(train, label);
        List<DataSet> dataSetList = dataSet.asList();
        Random rng = new Random(12345);
        Collections.shuffle(dataSetList, rng);
        return new ListDataSetIterator(dataSetList, batchSize);
    }

    private static INDArray loadData (File file) {
        INDArray ndarray = null;
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
            String line = reader.readLine();
            int rows = Integer.parseInt(line.split(",")[0]);
            int columns = Integer.parseInt(line.split(",")[1]);
            ndarray = Nd4j.create(rows, columns);
            int index = 0;
            while ((line = reader.readLine()) != null) {
                double[] array = Arrays.stream(line.split(",")).map(Double::parseDouble).mapToDouble(d -> d).toArray();
                for (int i = 0; i < array.length; i++) ndarray.put(index, i, Nd4j.scalar(array[i]));
                index++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return ndarray;
    }
}
