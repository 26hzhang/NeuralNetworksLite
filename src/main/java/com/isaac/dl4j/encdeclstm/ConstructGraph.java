package com.isaac.dl4j.encdeclstm;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Map;

class ConstructGraph {

    /** this is purely empirical, affects performance and VRAM requirement */
    private static final int HIDDEN_LAYER_WIDTH = 512;
    /** one-hot vectors will be embedded to more dense vectors with this width */
    private static final int EMBEDDING_WIDTH = 128;

    private static final int TBPTT_SIZE = 25;
    private static final double LEARNING_RATE = 1e-1;
    private static final double RMS_DECAY = 0.95;

    /**
     * Configure and initialize the computation graph. This is done once in the
     * beginning to prepare the computation graph for training.
     */
    static ComputationGraph createComputationGraph (Map<String, Double> dict) {
        final NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .iterations(1)
                .learningRate(LEARNING_RATE)
                .rmsDecay(RMS_DECAY)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .miniBatch(true)
                .updater(Updater.RMSPROP)
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer);

        final ComputationGraphConfiguration.GraphBuilder graphBuilder = builder.graphBuilder()
                .addInputs("inputLine", "decoderInput")
                .setInputTypes(InputType.recurrent(dict.size()), InputType.recurrent(dict.size()))
                .addLayer("embeddingEncoder",
                        new EmbeddingLayer.Builder()
                                .nIn(dict.size())
                                .nOut(EMBEDDING_WIDTH)
                                .build(),
                        "inputLine")
                .addLayer("encoder",
                        new GravesLSTM.Builder()
                                .nIn(EMBEDDING_WIDTH)
                                .nOut(HIDDEN_LAYER_WIDTH)
                                .activation(Activation.TANH)
                                .gateActivationFunction(Activation.HARDSIGMOID)
                                .build(),
                        "embeddingEncoder")
                .addVertex("thoughtVector",
                        new LastTimeStepVertex("inputLine"),
                        "encoder")
                .addVertex("dup",
                        new DuplicateToTimeSeriesVertex("decoderInput"),
                        "thoughtVector")
                .addVertex("merge",
                        new MergeVertex(),
                        "decoderInput",
                        "dup")
                .addLayer("decoder",
                        new GravesLSTM.Builder()
                                .nIn(dict.size() + HIDDEN_LAYER_WIDTH)
                                .nOut(HIDDEN_LAYER_WIDTH)
                                .activation(Activation.TANH)
                                .gateActivationFunction(Activation.HARDSIGMOID) // always be a (hard) sigmoid function
                                .build(),
                        "merge")
                .addLayer("output",
                        new RnnOutputLayer.Builder()
                                .nIn(HIDDEN_LAYER_WIDTH)
                                .nOut(dict.size())
                                .activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.MCXENT) // multi-class cross entropy
                                .build(),
                        "decoder")
                .setOutputs("output")
                .backpropType(BackpropType.Standard) // why not BackpropType.TruncatedBPTT
                .tBPTTForwardLength(TBPTT_SIZE)
                .tBPTTBackwardLength(TBPTT_SIZE)
                .pretrain(false)
                .backprop(true);

        ComputationGraph net = new ComputationGraph(graphBuilder.build());
        net.init();
        return net;
    }
}
