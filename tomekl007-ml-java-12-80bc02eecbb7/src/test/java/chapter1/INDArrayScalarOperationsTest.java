package chapter1;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import static org.nd4j.linalg.ops.transforms.Transforms.ceil;
import static org.nd4j.linalg.ops.transforms.Transforms.floor;
import static org.nd4j.linalg.ops.transforms.Transforms.round;
import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;

public class INDArrayScalarOperationsTest {

  private static INDArray SIX_BY_THREE_RANDOM = Nd4j.rand(new int[]{6, 3});
  private static INDArray TWO_BY_THREE_ONES = Nd4j.ones(2, 3);
  private static INDArray SIX_BY_THREE_ADD_TEN = Nd4j.create(new int[]{6, 3}).add(10);

  @Test
  public void subtractVectors() {

    INDArray vectorAdd = SIX_BY_THREE_RANDOM.add(SIX_BY_THREE_ADD_TEN);
    System.out.println("Vector add " + vectorAdd);
    INDArray vectorSubtract = SIX_BY_THREE_RANDOM.sub(SIX_BY_THREE_ADD_TEN);
    System.out.println("Vector subtract " + vectorSubtract);
  }

  @Test
  public void multiplyVectors() {
    INDArray vectorMultiply = SIX_BY_THREE_RANDOM.mul(SIX_BY_THREE_ADD_TEN);
    System.out.println("Vector multiply " + vectorMultiply);

  }

  @Test
  public void divideVectors() {
    INDArray vectorDivide = SIX_BY_THREE_RANDOM.div(SIX_BY_THREE_ADD_TEN);
    System.out.println("Vector divide" + vectorDivide);
  }

  @Test
  public void compareVectors() {
    boolean areArraysEquals1 = vectorEquals(SIX_BY_THREE_RANDOM, TWO_BY_THREE_ONES);
    boolean areArraysEquals2 = vectorEquals(SIX_BY_THREE_RANDOM, SIX_BY_THREE_RANDOM);
    System.out.println("Are arrays equals: 1. " + areArraysEquals1 + ", 2. " + areArraysEquals2);
  }

  @Test
  public void sqrtVectors() {
    INDArray sqrt = sqrt(SIX_BY_THREE_RANDOM);
    System.out.println("Vector square root: " + sqrt);
  }

  @Test
  public void ceilFloorAndRoundVectors() {
    INDArray ceil = ceil(SIX_BY_THREE_RANDOM);
    System.out.println("Vector ceil: " + ceil);

    INDArray floor = floor(SIX_BY_THREE_RANDOM);
    System.out.println("Vector floor: " + floor);

    INDArray round = round(SIX_BY_THREE_RANDOM);
    System.out.println("Vector round: " + round);
  }

  private static boolean vectorEquals(INDArray arr1, INDArray arr2) {
    return ArrayUtil.equals(arr1.data().asFloat(), arr2.data().asDouble());
  }
}
