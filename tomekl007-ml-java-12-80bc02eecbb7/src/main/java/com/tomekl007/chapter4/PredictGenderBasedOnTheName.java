package com.tomekl007.chapter4;

import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class PredictGenderBasedOnTheName {
  private static final ArrayList<String> LABELS = new ArrayList<String>() {{
    add("M");
    add("F");
  }};
  private String filePath;

  public static void main(String args[]) throws IOException {

    PredictGenderBasedOnTheName dg = new PredictGenderBasedOnTheName();
    dg.filePath = new ClassPathResource("PredictGender/Data").getFile().getAbsolutePath();
    System.out.println(dg.filePath);
    dg.startTraining();
  }

  public void startTraining() {
    int seed = 123456;
    double learningRate = 0.005;
    int batchSize = 100;
    int nEpochs = 10;
    int numInputs = 0;
    int numOutputs = 0;
    int numHiddenNodes = 0;

    try (LabeledGenderFromFileLineReader rr = new LabeledGenderFromFileLineReader(LABELS)) {

      rr.initialize(new FileSplit(new File(this.filePath)));


      numInputs = rr.maxLengthName * 5;  // multiplied by 5 as for each letter we use five binary digits like 00000
      numOutputs = 2;
      numHiddenNodes = 2 * numInputs + numOutputs;


      LabeledGenderFromFileLineReader rr1 = new LabeledGenderFromFileLineReader(LABELS);
      rr1.initialize(new FileSplit(new File(this.filePath)));

      DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, numInputs, 2);
      DataSetIterator testIter = new RecordReaderDataSetIterator(rr1, batchSize, numInputs, 2);

      MultiLayerConfiguration conf = configureNeuralNetwork(seed, learningRate, numInputs, numOutputs, numHiddenNodes);

      MultiLayerNetwork model = new MultiLayerNetwork(conf);
      model.init();

      configureUIServer(model);

      performTraining(nEpochs, testIter, model);


      performCrossValidation(numOutputs, trainIter, model);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  private void performCrossValidation(int numOutputs, DataSetIterator testIter, MultiLayerNetwork model) {
    System.out.println("Evaluate model....");
    Evaluation eval = new Evaluation(numOutputs);
    while (testIter.hasNext()) {
      DataSet t = testIter.next();
      INDArray features = t.getFeatures();
      INDArray lables = t.getLabels();
      INDArray predicted = model.output(features, false);

      eval.eval(lables, predicted);

    }
    System.out.println(eval.stats());
  }

  private void performTraining(int nEpochs, DataSetIterator trainIter, MultiLayerNetwork model) {
    for (int n = 0; n < nEpochs; n++) {
      while (trainIter.hasNext()) {
        model.fit(trainIter.next());
      }
      trainIter.reset();
    }
  }

  private void configureUIServer(MultiLayerNetwork model) {
    UIServer uiServer = UIServer.getInstance();
    StatsStorage statsStorage = new InMemoryStatsStorage();
    uiServer.attach(statsStorage);
    model.setListeners(new StatsListener(statsStorage));
  }

  private MultiLayerConfiguration configureNeuralNetwork(int seed, double learningRate, int numInputs, int numOutputs, int numHiddenNodes) {
    return new NeuralNetConfiguration.Builder()
        .seed(seed)
        .biasInit(1)
        .l2(1e-4)
        .updater(new Nesterovs(learningRate, 0.9))
        .list()
        .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.RELU)
            .build())
        .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.RELU)
            .build())
        .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.SOFTMAX)
            .nIn(numHiddenNodes).nOut(numOutputs).build())
        .build();
  }
}
