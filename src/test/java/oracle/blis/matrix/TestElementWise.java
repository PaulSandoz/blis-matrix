package oracle.blis.matrix;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;

public class TestElementWise {

    @Test
    public void testUnary() {
        var m = FloatMatrix.newFloatMatrix(5, 3);
        m.setEach((i, j, v) -> j * m.rows() + i);
        m.print();
//    0.00 5.00 10.00
//    1.00 6.00 11.00
//    2.00 7.00 12.00
//    3.00 8.00 13.00
//    4.00 9.00 14.00

        // We don't need a method for each expression or operation
        // We can use a lambda expression

        // Negate all the elements of m
        m.elementwise(v -> -v);
        m.print();
//    -0.00 -5.00 -10.00
//    -1.00 -6.00 -11.00
//    -2.00 -7.00 -12.00
//    -3.00 -8.00 -13.00
//    -4.00 -9.00 -14.00
    }


    @Test
    public void testBinarySameDimensions() {
        var m = FloatMatrix.newFloatMatrix(5, 3);
        m.setEach((i, j, v) -> j * m.rows() + i);
        m.print();
//    0.00 5.00 10.00
//    1.00 6.00 11.00
//    2.00 7.00 12.00
//    3.00 8.00 13.00
//    4.00 9.00 14.00

        // Same dimensions
        var n = FloatMatrix.newFloatMatrix(5, 3);
        n.setEach((i, j, v) -> j * m.rows() + i);
        n.print();
//    0.00 5.00 10.00
//    1.00 6.00 11.00
//    2.00 7.00 12.00
//    3.00 8.00 13.00
//    4.00 9.00 14.00

        // Multiply the elements of m and n at the same position, and store back in that position
        m.elementwise(n, (a, b) -> a * b);
        m.print();
//    0.00 5.00 10.00     0.00  5.00 10.00      0.00 25.00 100.00
//    1.00 6.00 11.00     1.00  6.00 11.00      1.00 36.00 121.00
//    2.00 7.00 12.00  *  2.00  7.00 12.00  =   4.00 49.00 144.00
//    3.00 8.00 13.00     3.00  8.00 13.00      9.00 64.00 169.00
//    4.00 9.00 14.00     4.00  9.00 14.00     16.00 81.00 196.00
    }


    @Test
    public void testBinaryBroadcastScalar() {
        var m = FloatMatrix.newFloatMatrix(5, 3);
        m.setEach((i, j, v) -> j + i * m.columns());
        m.print();
//    0.00 5.00 10.00
//    1.00 6.00 11.00
//    2.00 7.00 12.00
//    3.00 8.00 13.00
//    4.00 9.00 14.00

        // Broadcast scalar
        var scalar = FloatMatrix.newFloatMatrix(1, 1);
        scalar.setEach((i, j, v) -> -1);
        scalar.print();
//   -1.00

        m.elementwise(scalar, (a, b) -> a * b);
        m.print();
//    0.00 5.00 10.00     -1.00                  -0.00  -1.00  -2.00
//    1.00 6.00 11.00                            -3.00  -4.00  -5.00
//    2.00 7.00 12.00  *                     =   -6.00  -7.00  -8.00
//    3.00 8.00 13.00                            -9.00 -10.00 -11.00
//    4.00 9.00 14.00                           -12.00 -13.00 -14.00
// as if
//    0.00 5.00 10.00     -1.00 -1.00 -1.00      -0.00  -1.00  -2.00
//    1.00 6.00 11.00     -1.00 -1.00 -1.00      -3.00  -4.00  -5.00
//    2.00 7.00 12.00  *  -1.00 -1.00 -1.00  =   -6.00  -7.00  -8.00
//    3.00 8.00 13.00     -1.00 -1.00 -1.00      -9.00 -10.00 -11.00
//    4.00 9.00 14.00     -1.00 -1.00 -1.00     -12.00 -13.00 -14.00
    }


    @Test
    public void testBinaryBroadcastRowVector() {
        var m = FloatMatrix.newFloatMatrix(5, 3);
        m.setEach((i, j, v) -> j * m.rows() + i);
        m.print();
//    0.00 5.00 10.00
//    1.00 6.00 11.00
//    2.00 7.00 12.00
//    3.00 8.00 13.00
//    4.00 9.00 14.00

        var rowVector = FloatMatrix.newFloatMatrix(1, 3);
        rowVector.setEach((i, j, v) -> j);
        rowVector.print();
//    0.00 1.00 2.00

        m.elementwise(rowVector, (a, b) -> a * b);
        m.print();
//    0.00 5.00 10.00     0.00 1.00 2.00     0.00 5.00 20.00
//    1.00 6.00 11.00                        0.00 6.00 22.00
//    2.00 7.00 12.00  *                  =  0.00 7.00 24.00
//    3.00 8.00 13.00                        0.00 8.00 26.00
//    4.00 9.00 14.00                        0.00 9.00 28.00

//    0.00 5.00 10.00     0.00 1.00 2.00     0.00 5.00 20.00
//    1.00 6.00 11.00     0.00 1.00 2.00     0.00 6.00 22.00
//    2.00 7.00 12.00  *  0.00 1.00 2.00  =  0.00 7.00 24.00
//    3.00 8.00 13.00     0.00 1.00 2.00     0.00 8.00 26.00
//    4.00 9.00 14.00     0.00 1.00 2.00     0.00 9.00 28.00
    }


    @Test
    public void testBinaryBroadcastColumnVector() {
        var m = FloatMatrix.newFloatMatrix(5, 3);
        m.setEach((i, j, v) -> j * m.rows() + i);
        m.print();
//    0.00 5.00 10.00
//    1.00 6.00 11.00
//    2.00 7.00 12.00
//    3.00 8.00 13.00
//    4.00 9.00 14.00

        var columnVector = FloatMatrix.newFloatMatrix(5, 1);
        columnVector.setEach((i, j, v) -> i);
        columnVector.print();
//    0.00
//    1.00
//    2.00
//    3.00
//    4.00

        m.elementwise(columnVector, (a, b) -> a * b);
        m.print();
//    0.00 5.00 10.00     0.00                0.00  0.00  0.00
//    1.00 6.00 11.00     1.00                1.00  6.00 11.00
//    2.00 7.00 12.00  *  2.00            =   4.00 14.00 24.00
//    3.00 8.00 13.00     3.00                9.00 24.00 39.00
//    4.00 9.00 14.00     4.00               16.00 36.00 56.00

//    0.00 5.00 10.00     0.00 0.00 0.00      0.00  0.00  0.00
//    1.00 6.00 11.00     1.00 1.00 1.00      1.00  6.00 11.00
//    2.00 7.00 12.00  *  2.00 2.00 2.00  =   4.00 14.00 24.00
//    3.00 8.00 13.00     3.00 3.00 3.00      9.00 24.00 39.00
//    4.00 9.00 14.00     4.00 4.00 4.00     16.00 36.00 56.00
    }

    static class Accelerator {}

    static class MatrixScope implements AutoCloseable {
        static MatrixScope ofAccelerator(Accelerator a) {
            return null;
        }

        @Override
        public void close() {

        }

    }

    FloatMatrix columnMean(FloatMatrix A) {
        var mean = Matrix.newFloatMatrix(1, A.columns());
        // Column reduction producing a row vector
        mean.reductionColumn(mean, (a, b) -> a + b);
        // Unary division
        mean.elementwise(e -> e / A.rows());
        return mean;
    }

    FloatMatrix columnStd(FloatMatrix A, boolean isBiasCorrected) {
        var mean = columnMean(A);
        FloatMatrix std = Matrix.newFloatMatrix(1, A.columns());
        float x = isBiasCorrected ? A.rows() - 1.0f : A.rows();
        // Accumulated reduction
        A.reductionColumn(std,
                (j) -> {
                    var acc = new float[2];
                    acc[1] = mean.get(0, j);
                    return acc;
                },
                (acc, v) -> {
                    float d = v - acc[1];
                    acc[0] += d * d;
                    return acc;
                },
                acc -> (float) Math.sqrt(acc[0] / x)
        );
        return std;
    }
}