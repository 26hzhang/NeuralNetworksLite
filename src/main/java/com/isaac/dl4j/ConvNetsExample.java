package com.isaac.dl4j;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * Created by zhanghao on 21/6/17.
 * @author ZHANG HAO
 * A practice of build simple Convolutional Neural Networks via DL4J
 */
public class ConvNetsExample {
    public static final Logger log = LoggerFactory.getLogger(ConvNetsExample.class);

    public static void main (String[] args) throws IOException {
        // pre-defined constant
        int nChannels = 1; // MNIST dataset contains gray images
        int epochs = 1;
        int iterations = 1;
        int seed = 12345;
        int numClasses = 10; // ten different classes in MNIST dataset
        int batchSize = 64;
        //load MNIST Data for testing
        log.info("load data...");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize,true, seed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,false, seed);

        // construct model
        log.info("construct model");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.01)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .momentum(0.9)
                .regularization(true)
                .l2(0.0005)
                .list()
                .layer(0, new ConvolutionLayer.Builder()
                        .nIn(nChannels)
                        .nOut(20)
                        .kernelSize(5, 5)
                        .stride(2, 2)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new ConvolutionLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(2, new SubsamplingLayer.Builder()
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(3, new ConvolutionLayer.Builder()
                        .kernelSize(2, 2)
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(4, new SubsamplingLayer.Builder()
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(1, 1)
                        .build())
                .layer(5, new DenseLayer.Builder()
                        .activation(Activation.LEAKYRELU)
                        .nOut(500)
                        .build())
                .layer(6, new OutputLayer.Builder()
                        .nOut(numClasses)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28, 28, 1))
                .backprop(true)
                .pretrain(false)
                .build();
        MultiLayerNetwork networks = new MultiLayerNetwork(conf);
        networks.init();

        log.info("Train model....");
        networks.setListeners(new ScoreIterationListener(1));
        for( int i=0; i<epochs; i++ ) {
            networks.fit(mnistTrain);
            log.info("*** Completed epoch {} ***", i);
            log.info("Evaluate model....");
            Evaluation eval = new Evaluation(numClasses);
            while(mnistTest.hasNext()){
                DataSet ds = mnistTest.next();
                INDArray output = networks.output(ds.getFeatureMatrix(), false);
                eval.eval(ds.getLabels(), output);
            }
            log.info(eval.stats());
            mnistTest.reset();
        }
        log.info("Example finished...");
    }
}
/*
Activation.IDENTITY
==========================Scores========================================
 Accuracy:        0.9788
 Precision:       0.9787
 Recall:          0.9789
 F1 Score:        0.9788
========================================================================

Activation.RELU
==========================Scores========================================
 Accuracy:        0.9779
 Precision:       0.9779
 Recall:          0.978
 F1 Score:        0.9779
========================================================================
 */