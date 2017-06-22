package com.isaac.dl4j.houseprices;

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
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

/**
 * Created by zhanghao on 22/6/17.
 * @author ZHANG HAO
 */
public class Main {
    public static final Logger log = LoggerFactory.getLogger(Main.class);

    public static void main (String[] args) throws IOException {
        File trainFile = new ClassPathResource("House_Prices/train_pca.txt").getFile();
        File testFile = new ClassPathResource("House_Prices/test_pca.txt").getFile();
        File labelFile = new ClassPathResource("House_Prices/label.txt").getFile();
        INDArray train = DataLoader.loadData(trainFile);
        //System.out.println(train);
        INDArray test = DataLoader.loadData(testFile);
        INDArray label = DataLoader.loadData(labelFile);

        int epochs = 100;

        int nIn = train.columns();
        int nOut = 1;

        MultiLayerNetwork model = constructModel(nIn, nOut);
        model.init();
        //model.setListeners(new ScoreIterationListener(100));
        for (int epoch = 0; epoch < epochs; epoch++) {
            //for (int i = 0; i < train.rows(); i++) {
               // model.fit(train.getRow(i), label.getRow(i));
            //}
            model.fit(train, label);
        }
        for (int i = 0; i < test.rows(); i++) {
            System.out.println(Math.pow(Math.E, model.output(test.getRow(i), false).getDouble(0)));
        }
    }

    public static MultiLayerNetwork constructModel (int nIn, int nOut) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(1234)
                .iterations(1)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .momentum(0.9)
                .learningRate(0.1)
                //.lrPolicyDecayRate(0.01)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(nIn)
                        .nOut(50)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(50)
                        .nOut(20)
                        .activation(Activation.TANH)
                        .build())
                .layer(2, new OutputLayer.Builder()
                        .nIn(20)
                        .nOut(nOut)
                        .activation(Activation.IDENTITY)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .backprop(true)
                .pretrain(false)
                .build();
        return new MultiLayerNetwork(conf);
    }
}
