package com.isaac.dl4j.kaggle.houseprices;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.*;
import java.util.*;

/**
 * Created by zhanghao on 22/6/17.
 * @author ZHANG HAO
 */
public class DataLoader {

    public static DataSetIterator getTrainingData(int batchSize){
        File trainFile = null;
        File labelFile = null;
        try {
            trainFile = new ClassPathResource("House_Prices/train_pca.txt").getFile();
            labelFile = new ClassPathResource("House_Prices/label.txt").getFile();
        } catch (IOException e) {
            e.printStackTrace();
        }
        INDArray train = DataLoader.loadData(trainFile);
        INDArray label = DataLoader.loadData(labelFile);
        DataSet dataSet = new DataSet(train, label);
        List<DataSet> listDs = dataSet.asList();
        Random rng = new Random(12345);
        Collections.shuffle(listDs,rng);
        return new ListDataSetIterator(listDs,batchSize);
    }

    public static INDArray loadData (File file) {
        INDArray ndarray = null;
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
            String line = reader.readLine();
            int rows = Integer.parseInt(line.split(",")[0]);
            int columns = Integer.parseInt(line.split(",")[1]);
            ndarray = Nd4j.create(rows, columns);
            int index = 0;
            while ((line = reader.readLine()) != null) {
                double[] array = Arrays.stream(line.split(",")).map(Double::parseDouble).mapToDouble(d -> d.doubleValue()).toArray();
                for (int i = 0; i < array.length; i++) {
                    ndarray.put(index, i, Nd4j.scalar(array[i]));
                }
                index++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return ndarray;
    }

}
