package nd4j.isaac.neuralnetworks;

import nd4j.isaac.initialization.Activation;
import nd4j.isaac.initialization.Weight;
import nd4j.isaac.layers.DenoisingAutoencoder;
import nd4j.isaac.layers.DenseLayer;
import nd4j.isaac.layers.OutputLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by zhanghao on 20/6/17.
 * @author ZHANG HAO
 */
public class StackedDenoisingAutoencoder {
    public int nIn;
    public int[] hiddenLayerSizes;
    public int nOut;
    public Random rng;
    public DenoisingAutoencoder[] daLayers;
    public DenseLayer[] hiddenLayers;
    public OutputLayer outputLayer;

    private int nLayers;

    public StackedDenoisingAutoencoder(int nIn, int[] hiddenLayerSizes, int nOut, Random rng) {
        this.nIn = nIn;
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.nOut = nOut;
        this.nLayers = this.hiddenLayerSizes.length;
        this.rng = rng == null ? new Random(12345) : rng;
        this.daLayers = new DenoisingAutoencoder[nLayers];
        this.hiddenLayers = new DenseLayer[nLayers];
        for (int i = 0; i < nLayers; i++) {
            int nIn_ = i == 0 ? nIn : hiddenLayerSizes[i - 1];
            hiddenLayers[i] = new DenseLayer(nIn_, hiddenLayerSizes[i], Weight.UNIFORM, Activation.Sigmoid, this.rng);
            daLayers[i] = new DenoisingAutoencoder(nIn_, hiddenLayerSizes[i], hiddenLayers[i].W, hiddenLayers[i].b, null, Activation.Sigmoid, this.rng);
        }
        this.outputLayer = new OutputLayer(hiddenLayerSizes[nLayers - 1], nOut, Weight.ZERO, Activation.SoftMax, this.rng);
    }

    public void preTrain(List<INDArray> X, int minibatchSize, int minibatch_N, int epochs, double learningRate, double corruptionLevel) {
        for (int layer = 0; layer < nLayers; layer++) {
            for (int epoch = 0; epoch < epochs; epoch++) {
                for (int batch = 0; batch < minibatch_N; batch++) {
                    INDArray X_ = Nd4j.create(new double[minibatchSize * nIn], new int[] { minibatchSize, nIn });
                    INDArray prevLayerX_;
                    // Set input data for current layer
                    if (layer == 0) {
                        X_ = X.get(batch);
                    } else {
                        prevLayerX_ = X_;
                        X_ = hiddenLayers[layer - 1].output(prevLayerX_);
                    }
                    daLayers[layer].train(X_, minibatchSize, learningRate, corruptionLevel);
                }
            }
        }
    }

    public void finetune(INDArray X, INDArray T, int minibatchSize, double learningRate) {
        List<INDArray> layerInputs = new ArrayList<>(nLayers + 1);
        layerInputs.add(X);
        INDArray Z = X.dup();
        INDArray dY;
        // forward hidden layers
        for (int layer = 0; layer < nLayers; layer++) {
            INDArray x_; // layer input
            INDArray Z_;
            if (layer == 0) {
                x_ = X;
            } else {
                x_ = Z;
            }
            Z_ = hiddenLayers[layer].forward(x_);
            Z = Z_;
            layerInputs.add(Z.dup());
        }
        // forward & backward output layer
        dY = outputLayer.train(Z, T, minibatchSize, learningRate);
        // backward hidden layers
        INDArray Wprev;
        INDArray dZ = Z.dup();
        for (int layer = nLayers - 1; layer >= 0; layer--) {
            if (layer == nLayers - 1) {
                Wprev = outputLayer.W;
            } else {
                Wprev = hiddenLayers[layer + 1].W;
                dY = dZ.dup();
            }
            dZ = hiddenLayers[layer].backward(layerInputs.get(layer), layerInputs.get(layer + 1), dY, Wprev, minibatchSize, learningRate);
        }
    }

    public INDArray predict(INDArray x) {
        INDArray z = x.dup();
        for (int layer = 0; layer < nLayers; layer++) {
            INDArray x_;
            if (layer == 0) x_ = x;
            else x_ = z.dup();
            z = hiddenLayers[layer].forward(x_);
        }
        return outputLayer.predict(z);
    }
}
