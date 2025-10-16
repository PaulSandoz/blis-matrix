package oracle.blis.matrix;

import jdk.incubator.vector.Float16;
import org.junit.jupiter.api.Test;

public class TestFloat16 {

    @Test
    public void testElementalAccessUnary() {
        var m = Float16Matrix.newFloat16Matrix(10, 4);
        m.setEach((i, j, v) -> Float16.valueOf(j + i * m.columns()));
        m.print();

        m.elementwise(v -> Float16.negate(v));
        m.print();
    }


    @Test
    public void testBinaryElementwise() {
        var m = Float16Matrix.newFloat16Matrix(10, 4);
        m.setEach((i, j, v) -> Float16.valueOf(j + i * m.columns()));
        m.print();

        // Broadcast scalar
        var scalar = DoubleMatrix.newFloat16Matrix(1, 1);
        scalar.setEach((i, j, v) -> Float16.valueOf(-1));
        m.elementwise(scalar, (a, b) -> Float16.multiply(a, b));
        m.print();

        // Broadcast vector
        var rowVector = DoubleMatrix.newFloat16Matrix(1, 4);
        rowVector.setEach((i, j, v) -> Float16.valueOf(j % 2 == 0 ? 1 : -1));
        m.elementwise(rowVector, (a, b) -> Float16.multiply(a, b));
        m.print();

        // Broadcast column
        var columnVector = DoubleMatrix.newFloat16Matrix(10, 1);
        columnVector.setEach((i, j, v) -> Float16.valueOf(i % 2 == 0 ? 1 : -1));
        m.elementwise(columnVector, (a, b) -> Float16.multiply(a, b));
        m.print();

        // Same dimensions
        var n = DoubleMatrix.newFloat16Matrix(10, 4);
        m.setEach((i, j, v) -> Float16.valueOf(j + i * m.columns()));
        m.elementwise(n, (a, b) -> Float16.add(a, b));
        m.print();
    }

    Float16Matrix columnMean(Float16Matrix A) {
        var mean = Matrix.newFloat16Matrix(1, A.columns());
        // Column reduction producing a row vector
        mean.reductionColumn(mean, Float16::add);
        // Unary division
        mean.elementwise(e -> Float16.divide(e, Float16.valueOf(A.rows())));
        return mean;
    }

    Float16Matrix columnStd(Float16Matrix A, boolean isBiasCorrected) {
        var mean = columnMean(A);
        Float16Matrix std = Matrix.newFloat16Matrix(1, A.columns());
        float x = isBiasCorrected ? A.rows() - 1.0f : A.rows();
        // Accumulated reduction
        A.reductionColumn(std,
                (j) -> {
                    var acc = new float[2];
                    acc[1] = mean.get(0, j).floatValue();
                    return acc;
                },
                (acc, v) -> {
                    float d = v.floatValue() - acc[1];
                    acc[0] += d * d;
                    return acc;
                },
                acc -> Float16.valueOf(Math.sqrt(acc[0] / x))
        );
        return std;
    }

}
