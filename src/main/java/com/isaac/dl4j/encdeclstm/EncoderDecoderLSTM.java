package com.isaac.dl4j.encdeclstm;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;

import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.layers.recurrent.GravesLSTM;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class EncoderDecoderLSTM {
    /** filename of data corpus to learn */
    private static final String CORPUS_FILENAME = "movie_lines.txt";
    /** filename of the model */
    private static final String MODEL_FILENAME = "rnn_train.zip";
    /** filename of the previous version of the model (backup) */
    private static final String BACKUP_MODEL_FILENAME = "rnn_train.bak.zip";
    private static final int MINIBATCH_SIZE = 32;
    private static final Random rnd = new Random(new Date().getTime());
    private static final long SAVE_EACH_MS = TimeUnit.MINUTES.toMillis(5); // save the model with this period
    private static final long TEST_EACH_MS = TimeUnit.MINUTES.toMillis(1); // test the model with this period
    /** maximum line length in tokens */
    private static final int ROW_SIZE = 40;
    /**
     * The delay between invocations of {@link System#gc()} in
     * milliseconds. If VRAM is being exhausted, reduce this value. Increase
     * this value to yield better performance.
     */
    private static final int GC_WINDOW = 2000;
    /** see CorpusIterator */
    private static final int MACROBATCH_SIZE = 20;
    /** The computation graph model. */
    private static ComputationGraph net;

    public static void main(String[] args) throws IOException {
        Nd4j.getMemoryManager().setAutoGcWindow(GC_WINDOW);
        // create dictionaries
        Dictionaries.createDictionary(CORPUS_FILENAME, ROW_SIZE);
        // try to load network file
        File networkFile = new File(obtainFilePath(MODEL_FILENAME));
        int offset = 0;
        if (networkFile.exists()) {
            System.out.println("Loading the existing network...");
            net = ModelSerializer.restoreComputationGraph(networkFile);
            System.out.print("Enter d to start dialog or a number to continue training from that minibatch: ");
            String input;
            try (Scanner scanner = new Scanner(System.in)) {
                input = scanner.nextLine();
                if (input.toLowerCase().equals("d")) {
                    startDialog(scanner);
                } else {
                    offset = Integer.valueOf(input);
                    test();
                }
            }
        } else {
            System.out.println("Creating a new network...");
            net = ConstructGraph.createComputationGraph(Dictionaries.dict);
        }
        System.out.println("Number of parameters: " + net.numParams());
        net.setListeners(new ScoreIterationListener(1));
        train(networkFile, offset);
    }

    private static void train(File networkFile, int offset) throws IOException {
        long lastSaveTime = System.currentTimeMillis();
        long lastTestTime = System.currentTimeMillis();
        CorpusIterator logsIterator = new CorpusIterator(Dictionaries.corpus, MINIBATCH_SIZE, MACROBATCH_SIZE, Dictionaries.dict.size(), ROW_SIZE);
        for (int epoch = 1; epoch < 10000; ++epoch) {
            System.out.println("Epoch " + epoch);
            if (epoch == 1) logsIterator.setCurrentBatch(offset);
            else logsIterator.reset();
            int lastPerc = 0;
            while (logsIterator.hasNextMacrobatch()) {
                net.fit(logsIterator);
                logsIterator.nextMacroBatch();
                System.out.println("Batch = " + logsIterator.batch());
                int newPerc = (logsIterator.batch() * 100 / logsIterator.totalBatches());
                if (newPerc != lastPerc) {
                    System.out.println("Epoch complete: " + newPerc + "%");
                    lastPerc = newPerc;
                }
                if (System.currentTimeMillis() - lastSaveTime > SAVE_EACH_MS) {
                    saveModel(networkFile);
                    lastSaveTime = System.currentTimeMillis();
                }
                if (System.currentTimeMillis() - lastTestTime > TEST_EACH_MS) {
                    test();
                    lastTestTime = System.currentTimeMillis();
                }
            }
        }
    }

    private static void startDialog(Scanner scanner) throws IOException {
        System.out.println("Dialog started.");
        while (true) {
            System.out.print("In> ");
            // input line is appended to conform to the corpus format
            String line = "1 +++$+++ u11 +++$+++ m0 +++$+++ WALTER +++$+++ " + scanner.nextLine() + "\n";
            CorpusProcessor dialogProcessor = new CorpusProcessor(new
                    ByteArrayInputStream(line.getBytes(StandardCharsets.UTF_8)), ROW_SIZE, false) {
                @Override
                protected void processLine(String lastLine) {
                    List<String> words = new ArrayList<>();
                    tokenizeLine(lastLine, words, true);
                    final List<Double> wordIdxs = wordsToIndexes(words);
                    if (!wordIdxs.isEmpty()) {
                        System.out.print("Got words: ");
                        for (Double idx : wordIdxs) {
                            System.out.print(Dictionaries.revDict.get(idx) + " ");
                        }
                        System.out.println();
                        System.out.print("Out> ");
                        output(wordIdxs, true);
                    }
                }
            };
            dialogProcessor.setDict(Dictionaries.dict);
            dialogProcessor.start();
        }
    }

    private static void saveModel(File networkFile) throws IOException {
        System.out.println("Saving the model...");
        File backup = new File(obtainFilePath(BACKUP_MODEL_FILENAME));
        if (networkFile.exists()) {
            if (backup.exists()) backup.delete();
            networkFile.renameTo(backup);
        }
        ModelSerializer.writeModel(net, networkFile, true);
        System.out.println("Done.");
    }

    private static void test() {
        System.out.println("======================== TEST ========================");
        int selected = rnd.nextInt(Dictionaries.corpus.size());
        List<Double> rowIn = new ArrayList<>(Dictionaries.corpus.get(selected));
        System.out.print("In: ");
        for (Double idx : rowIn) System.out.print(Dictionaries.revDict.get(idx) + " ");
        System.out.println();
        System.out.print("Out: ");
        output(rowIn, true);
        System.out.println("====================== TEST END ======================");
    }

    private static void output(List<Double> rowIn, boolean printUnknowns) {
        net.rnnClearPreviousState();
        Collections.reverse(rowIn);
        INDArray in = Nd4j.create(ArrayUtils.toPrimitive(rowIn.toArray(new Double[0])), new int[] { 1, 1, rowIn.size() });
        double[] decodeArr = new double[Dictionaries.dict.size()];
        decodeArr[2] = 1;
        INDArray decode = Nd4j.create(decodeArr, new int[] { 1, Dictionaries.dict.size(), 1 });
        net.feedForward(new INDArray[] { in, decode }, false);
        GravesLSTM decoder = (GravesLSTM) net.getLayer("decoder");
        Layer output = net.getLayer("output");
        GraphVertex mergeVertex = net.getVertex("merge");
        INDArray thoughtVector = mergeVertex.getInputs()[1];
        for (int row = 0; row < ROW_SIZE; ++row) {
            mergeVertex.setInputs(decode, thoughtVector);
            INDArray merged = mergeVertex.doForward(false);
            INDArray activateDec = decoder.rnnTimeStep(merged);
            INDArray out = output.activate(activateDec, false);
            double d = rnd.nextDouble();
            double sum = 0.0;
            int idx = -1;
            for (int s = 0; s < out.size(1); s++) {
                sum += out.getDouble(0, s, 0);
                if (d <= sum) {
                    idx = s;
                    if (printUnknowns || s != 0) System.out.print(Dictionaries.revDict.get((double) s) + " ");
                    break;
                }
            }
            if (idx == 1) break;
            double[] newDecodeArr = new double[Dictionaries.dict.size()];
            newDecodeArr[idx] = 1;
            decode = Nd4j.create(newDecodeArr, new int[] { 1, Dictionaries.dict.size(), 1 });
        }
        System.out.println();
    }

    static String obtainFilePath(String path) {
        return "src/main/resources/encdec/" + path;
    }

}
