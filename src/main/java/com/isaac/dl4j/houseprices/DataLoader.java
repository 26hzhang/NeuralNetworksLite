package com.isaac.dl4j.houseprices;

import com.isaac.nd4j.utils.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by zhanghao on 22/6/17.
 * @author ZHANG HAO
 */
public class DataLoader {

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
