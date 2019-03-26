package chapter1;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class INDArrayCreationTest {

  private static int NUMBER_OF_ROWS = 5;
  private static int NUMBER_OF_COLUMNS = 6;
  private static int[] shape = new int[]{NUMBER_OF_ROWS, NUMBER_OF_COLUMNS};

  @Test
  public void shouldCreateArrayFromFactoryMethods() {
    INDArray allZeros = Nd4j.zeros(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS);
    System.out.println("Nd4j.zeros(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS)");
    System.out.println(allZeros);

    INDArray allOnes = Nd4j.ones(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS);
    System.out.println("\nNd4j.ones(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS)");
    System.out.println(allOnes);

    INDArray allTens = Nd4j.valueArrayOf(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS, 10.0);
    System.out.println("\nNd4j.valueArrayOf(NUMBER_OF_ROWS, NUMBER_OF_COLUMNS, 10.0)");
    System.out.println(allTens);

  }

  @Test
  public void shouldCreateArrayFromPrimitiveTypeArray() {
    double[] vectorDouble = new double[]{1, 2, 3};
    INDArray rowVector = Nd4j.create(vectorDouble);
    System.out.println("rowVector:              " + rowVector);
    System.out.println("rowVector.shape():      " + Arrays.toString(rowVector.shape()));

    INDArray columnVector = Nd4j.create(vectorDouble, new int[]{3, 1});
    System.out.println("columnVector:           " + columnVector);
    System.out.println("columnVector.shape():   " + Arrays.toString(columnVector.shape()));

    double[][] matrixDouble = new double[][]{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}};
    INDArray matrix = Nd4j.create(matrixDouble);
    System.out.println("\nINDArray defined from double[][]:");
    System.out.println(matrix);


  }

  @Test
  public void shouldCreateRandomArray() {
    INDArray uniformRandom = Nd4j.rand(shape);
    System.out.println("\n\n\nUniform random array:");
    System.out.println(uniformRandom);
    System.out.println("Full precision of random value at position (0,0): " + uniformRandom.getDouble(0, 0));

    INDArray gaussianMeanZeroUnitVariance = Nd4j.randn(shape);
    System.out.println("\nN(0,1) random array:");
    System.out.println(gaussianMeanZeroUnitVariance);
  }

  @Test
  public void shouldCreateRepeatableArray() {

    long rngSeed = 12345;
    INDArray uniformRandom2 = Nd4j.rand(shape, rngSeed);
    INDArray uniformRandom3 = Nd4j.rand(shape, rngSeed);
    System.out.println("\nUniform random arrays with same fixed seed:");
    System.out.println(uniformRandom2);
    System.out.println();
    System.out.println(uniformRandom3);


  }

  @Test
  public void shouldCreateMoreThat3dArray() {
    //Of course, we aren't restricted to 2d. 3d or higher is easy:
    INDArray threeDimArray = Nd4j.ones(3, 4, 5);      //3x4x5 INDArray
    INDArray fourDimArray = Nd4j.ones(3, 4, 5, 6);     //3x4x5x6 INDArray
    INDArray fiveDimArray = Nd4j.ones(3, 4, 5, 6, 7);   //3x4x5x6x7 INDArray
    System.out.println("\n\n\nCreating INDArrays with more dimensions:");
    System.out.println("3d array shape:         " + Arrays.toString(threeDimArray.shape()));
    System.out.println("4d array shape:         " + Arrays.toString(fourDimArray.shape()));
    System.out.println("5d array shape:         " + Arrays.toString(fiveDimArray.shape()));

  }

  @Test
  public void shouldCombineArrays() {
    //We can create INDArrays by combining other INDArrays, too:
    INDArray rowVector1 = Nd4j.create(new double[]{1, 2, 3});
    INDArray rowVector2 = Nd4j.create(new double[]{4, 5, 6});

    INDArray vStack = Nd4j.vstack(rowVector1, rowVector2);      //Vertical stack:   [1,3]+[1,3] to [2,3]
    INDArray hStack = Nd4j.hstack(rowVector1, rowVector2);      //Horizontal stack: [1,3]+[1,3] to [1,6]
    System.out.println("\n\n\nCreating INDArrays from other INDArrays, using hstack and vstack:");
    System.out.println("vStack:\n" + vStack);
    System.out.println("hStack:\n" + hStack);
  }
}
