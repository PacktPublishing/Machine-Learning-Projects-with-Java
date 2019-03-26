package chapter1;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class INDArrayStatisticsTest {
  private static INDArray THREE_BY_TWO_RANDOM = Nd4j.rand(new int[]{3, 2});

  @Test
  public void shouldCalculateMeanOnDimensionZero() {
    INDArray mean = Nd4j.mean(THREE_BY_TWO_RANDOM, 0);
    System.out.println("Mean on dimension zero: " + mean);
  }

  @Test
  public void shouldCalculateSum() {
    Number sum = THREE_BY_TWO_RANDOM.sumNumber();
    System.out.println("Sum: " + sum);
  }

  @Test
  public void shouldCalculateMinAndMax() {
    Number min = THREE_BY_TWO_RANDOM.minNumber();
    System.out.println("Min: " + min);

    Number max = THREE_BY_TWO_RANDOM.maxNumber();
    System.out.println("Max: " + max);
  }


  @Test
  public void shouldCalculateVarianceAndStandardDeviation() {
    INDArray var = Nd4j.var(THREE_BY_TWO_RANDOM);
    System.out.println("Variance: " + var);

    INDArray std = Nd4j.std(THREE_BY_TWO_RANDOM, 1);
    System.out.println("Standard deviation: " + std);
  }
}

