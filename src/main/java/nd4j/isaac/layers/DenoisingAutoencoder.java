package nd4j.isaac.layers;

import nd4j.isaac.initialization.Activation;
import nd4j.isaac.initialization.ActivationFunction;
import nd4j.isaac.initialization.Weight;
import nd4j.isaac.initialization.WeightInitialization;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;
import java.util.function.Function;

/**
 * Created by zhanghao on 20/6/17.
 * @author  ZHANG HAO
 */
public class DenoisingAutoencoder {
    public int nVisible;
    public int nHidden;
    public INDArray W;
    public INDArray vbias;
    public INDArray hbias;
    public Random rng;
    private Function<INDArray, INDArray> activation; // normally, use Sigmoid

    public DenoisingAutoencoder (int nVisible, int nHidden, Weight weight, Activation activation, Random rng) {
        this.nVisible = nVisible;
        this.nHidden = nHidden;
        this.rng = rng == null ? new Random(12345) : rng;
        this.W = WeightInitialization.initialize(nVisible, nHidden, this.rng, weight);
        this.hbias = Nd4j.create(new double[nHidden], new int[] {nHidden, 1});
        this.vbias = Nd4j.create(new double[nVisible], new int[] {nVisible, 1});
        this.activation = ActivationFunction.activations(activation);
    }

    public DenoisingAutoencoder(int nVisible, int nHidden, INDArray W, INDArray hbias, INDArray vbias, Activation activation, Random rng) {
        this.nVisible = nVisible;
        this.nHidden = nHidden;
        this.rng = rng == null ? new Random(12345) : rng;
        this.W = W;
        this.hbias = hbias == null ? Nd4j.create(new double[nHidden], new int[]{nHidden, 1}) : hbias;
        this.vbias = vbias == null ? Nd4j.create(new double[nVisible], new int[] {nVisible, 1}) : vbias;
        this.activation = ActivationFunction.activations(activation);
    }

    public void train(INDArray X, int minibatchSize, double learningRate, double corruptionLevel) {
        // train with mini-batches
        // 1. add noise to original inputs
        INDArray corruptedInput = getCorruptedInput(X, corruptionLevel);
        // 2. encode
        INDArray z = getHiddenValues(corruptedInput);
        // 3. decode
        INDArray y = getReconstructedInput(z);
        // calculate gradients
        // update vbias
        INDArray v_ = X.sub(y);
        vbias.addi(v_.sum(0).transpose().mul(Nd4j.scalar(learningRate / minibatchSize)));
        // update hbias
        INDArray h_ = v_.mmul(W.transpose()).mul(z).mul(z.sub(Nd4j.scalar(1)).mul(Nd4j.scalar(-1.0)));
        hbias.addi(h_.sum(0).transpose().mul(Nd4j.scalar(learningRate / minibatchSize)));
        // update W
        W.addi(h_.transpose().mmul(corruptedInput).add(z.transpose().mmul(v_)).mul(Nd4j.scalar(learningRate / minibatchSize)));
    }

    public INDArray getCorruptedInput(INDArray x, double corruptionLevel) {
        INDArray corruptedInput = Nd4j.create(new double[x.rows() * x.columns()], new int[] { x.rows(), x.columns() });
        // add masking noise
        for (int i = 0; i < x.rows(); i++) {
            for (int j = 0; j < x.columns(); j++) {
                double rand_ = rng.nextDouble();
                if (rand_ < corruptionLevel) {
                    corruptedInput.put(i, j, Nd4j.scalar(0.0));
                } else {
                    corruptedInput.put(i, j, x.getDouble(i, j));
                }
            }
        }
        return corruptedInput;
    }

    public INDArray getHiddenValues(INDArray x) {
        return activation.apply(x.mmul(W.transpose()).addRowVector(hbias.transpose()));
    }

    public INDArray getReconstructedInput(INDArray z) {
        return activation.apply(z.mmul(W).addRowVector(vbias.transpose()));
    }

    public INDArray reconstruct(INDArray x) {
        return getReconstructedInput(getHiddenValues(x));
    }
}
