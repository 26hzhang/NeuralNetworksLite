package com.isaac.dl4j;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
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
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;

/**
 * Created by zhanghao on 21/6/17.
 * @author ZHANG HAO
 */
public class GlassClassification {
    public static final Logger log = LoggerFactory.getLogger(GlassClassification.class);

    public static void main(String[] args) throws IOException, InterruptedException {
        // source from: https://www.kaggle.com/uciml/glass
        int numLinesToSkip = 1;
        String delimiter = ",";
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("glass/glass.csv").getFile()));
        int labelIndex = 9;
        int numClasses = 7;
        int batchSize = 214; // totally, 214 data

        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
        DataSet allData = iterator.next();
        allData.shuffle();

        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.8);
        DataSet trainingData = testAndTrain.getTrain();

        DataSet testData = testAndTrain.getTest();

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);
        normalizer.transform(trainingData);
        normalizer.transform(testData);

        int seed = 123;
        int numInputs = 9;
        int iterations = 1000;
        int epochs = 1;

        log.info("Construct model...");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .momentum(0.9)
                .learningRate(0.2)
                .regularization(true)
                .l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(50)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nOut(100)
                        .activation(Activation.TANH)
                        .build())
                .layer(2, new OutputLayer.Builder()
                        .nOut(numClasses)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true)
                .pretrain(false)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));
        for (int epoch = 0; epoch < epochs; epoch++) {
            model.fit(trainingData);
            log.info("*** Completed epoch {} ***", epoch);
            log.info("Evaluate model....");
            Evaluation eval = new Evaluation(numClasses);
            INDArray output = model.output(testData.getFeatureMatrix());
            eval.eval(testData.getLabels(), output);
            log.info(eval.stats());
        }

    }

    /*public static void preProcess() throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(
            new ClassPathResource("glass-classification/glass.csv").getFile())));
        String line;
        List<String> list = new ArrayList<>();
        while ((line = reader.readLine()) != null) {
            if (line.contains("RI"))
                list.add(line);
            else {
                line = line.substring(0, line.lastIndexOf(",") + 1) +
                    (Integer.parseInt(line.substring(line.lastIndexOf(",") + 1)) - 1);
                list.add(line);
            }
        }
        reader.close();
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(
            new ClassPathResource("glass-classification/glass.csv").getFile())));
        for (String str : list) {
            writer.write(str + "\n");
        }
        writer.close();
    }*/
}
/*
Examples labeled as 0 classified by model as 0: 10 times
Examples labeled as 0 classified by model as 1: 4 times
Examples labeled as 1 classified by model as 1: 20 times
Examples labeled as 1 classified by model as 6: 1 times
Examples labeled as 2 classified by model as 0: 1 times
Examples labeled as 2 classified by model as 2: 1 times
Examples labeled as 5 classified by model as 5: 1 times
Examples labeled as 6 classified by model as 4: 1 times
Examples labeled as 6 classified by model as 6: 4 times

Warning: class 3 was never predicted by the model. This class was excluded from the average precision
Warning: class 3 has never appeared as a true label. This class was excluded from the average recall
Warning: class 4 has never appeared as a true label. This class was excluded from the average recall

==========================Scores========================================
 Accuracy:        0.8372
 Precision:       0.7571
 Recall:          0.7933
 F1 Score:        0.7748
========================================================================
 */