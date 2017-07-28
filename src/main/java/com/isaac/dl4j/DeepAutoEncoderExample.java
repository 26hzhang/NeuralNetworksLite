package com.isaac.dl4j;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ***** NOTE: This example has not been tuned. It requires additional work to produce sensible results *****
 *
 * @author Adam Gibson
 */
public class DeepAutoEncoderExample {

    private static Logger log = LoggerFactory.getLogger(DeepAutoEncoderExample.class);

    public static void main(String[] args) throws Exception {
        final int numRows = 28;
        final int numColumns = 28;
        int seed = 123;
        int numSamples = MnistDataFetcher.NUM_EXAMPLES;
        int batchSize = 1000;
        int iterations = 1;
        int listenerFreq = iterations / 5;

        log.info("Load data....");
        DataSetIterator iter = new MnistDataSetIterator(batchSize, numSamples, true);

	    //Get the DataSetIterators:
	    DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, seed);
	    DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, seed);

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .list()
                .layer(0, new RBM.Builder()
                        .nIn(numRows * numColumns)
		                .nOut(1000)
		                .lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE)
		                .build())
                .layer(1, new RBM.Builder()
		                .nIn(1000)
		                .nOut(500)
		                .lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE)
		                .build())
                .layer(2, new RBM.Builder()
		                .nIn(500)
		                .nOut(250)
		                .lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE)
		                .build())
                .layer(3, new RBM.Builder()
		                .nIn(250)
		                .nOut(100)
		                .lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE)
		                .build())
                .layer(4, new RBM.Builder()
		                .nIn(100)
		                .nOut(30)
		                .lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE)
		                .build()) //encoding stops
                .layer(5, new RBM.Builder()
		                .nIn(30)
		                .nOut(100)
		                .lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE)
		                .build()) //decoding starts
                .layer(6, new RBM.Builder()
		                .nIn(100)
		                .nOut(250)
		                .lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE)
		                .build())
                .layer(7, new RBM.Builder()
		                .nIn(250)
		                .nOut(500)
		                .lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE)
		                .build())
                .layer(8, new RBM.Builder()
		                .nIn(500)
		                .nOut(1000)
		                .lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE)
		                .build())
                .layer(9, new OutputLayer.Builder()
		                .lossFunction(LossFunctions.LossFunction.MSE)
		                .activation(Activation.SIGMOID)
		                .nIn(1000)
		                .nOut(numRows*numColumns)
		                .build())
                .pretrain(true)
		        .backprop(true)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new ScoreIterationListener(listenerFreq));

        log.info("Train model....");
        while(iter.hasNext()) {
            DataSet next = iter.next();
            model.fit(new DataSet(next.getFeatureMatrix(),next.getFeatureMatrix()));
        }
	    model.fit(mnistTrain);
	    log.info("done");

	    log.info("Evaluate model....");
	    Evaluation eval = new Evaluation(10); //create an evaluation object with 10 possible classes
	    while (mnistTest.hasNext()) {
		    DataSet next = mnistTest.next();
		    INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
		    eval.eval(next.getLabels(), output); //check the prediction against the true class
	    }
	    log.info(eval.stats());
    }
}