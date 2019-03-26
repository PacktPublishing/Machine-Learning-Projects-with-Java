package chapter1;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class INDArrayCreateMatrixTest {

  @Test
  public void shouldCreateMatrix() {
    int nRows = 3;
    int nColumns = 4;
    INDArray myArray = Nd4j.zeros(nRows, nColumns);

    System.out.println("Basic INDArray information:");
    System.out.println("Num. Rows:          " + myArray.rows());
    System.out.println("Num. Columns:       " + myArray.columns());
    System.out.println("Num. Dimensions:    " + myArray.rank());
    System.out.println("Shape:              " + Arrays.toString(myArray.shape()));
    System.out.println("Length:             " + myArray.length());

    System.out.println("\nArray Contents:\n" + myArray);

    System.out.println();
    System.out.println("size(0) == nRows:   " + myArray.size(0));
    System.out.println("size(1) == nCols:   " + myArray.size(1));
    System.out.println("Is a vector:        " + myArray.isVector());
    System.out.println("Is a scalar:        " + myArray.isScalar());
    System.out.println("Is a matrix:        " + myArray.isMatrix());
    System.out.println("Is a square matrix: " + myArray.isSquare());


  }

  @Test
  public void shouldPopulateArray() {
    int nRows = 4;
    int nColumns = 10;
    INDArray myArray = Nd4j.zeros(nRows, nColumns);
    double val0 = myArray.getDouble(0, 1);
    System.out.println("\nValue at (0,1):     " + val0);

    INDArray myArray2 = myArray.add(1.0);
    System.out.println("\nNew INDArray, after adding 1.0 to each entry:");
    System.out.println(myArray2);

    INDArray myArray3 = myArray2.mul(2.0);
    System.out.println("\nNew INDArray, after multiplying each entry by 2.0:");
    System.out.println(myArray3);

  }
}
