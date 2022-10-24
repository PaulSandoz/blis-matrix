/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */

package oracle.blis.matrix;

import oracle.blis.matrix.binding.blis_h;
import oracle.blis.matrix.binding.obj_t;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.ValueLayout;
import java.util.Objects;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

public final class DoubleMatrix extends Matrix<Double> {
    DoubleMatrix(SegmentAllocator allocator, MemorySegment obj, MemorySegment buffer) {
        super(allocator, obj, buffer);
    }

    public double get(long i, long j) {
        Objects.checkIndex(i, rows);
        Objects.checkIndex(j, columns);

        long index = linearIndex(i, j);
        return buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, index);
    }

    public void set(long i, long j, double v) {
        Objects.checkIndex(i, rows);
        Objects.checkIndex(j, columns);

        long index = linearIndex(i, j);
        buffer.setAtIndex(ValueLayout.JAVA_DOUBLE, index, v);
    }

    // Unary, update a in place
    public void elementwise(DoubleUnaryOperator o) {
        elementwise(this, o);
    }

    // Unary, update in r
    public void elementwise(Matrix<Double> r, DoubleUnaryOperator o) {
        DoubleMatrix result = cast(r);

        if (rows != result.rows || columns != result.columns) {
            throw new IllegalArgumentException("All matrices must have the same dimensions");
        }

        if (isZeroOffset() && r.isZeroOffset() &&
                o instanceof DoubleBroadcastKernel.UnaryBroadcastKernel k) {
            k.unaryBroadcast(buffer, r.buffer, rows * columns);
        } else {
            for (long j = 0; j < columns; j++) {
                for (long i = 0; i < rows; i++) {
                    long aIndex = linearIndex(i, j);
                    double aE = buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, aIndex);
                    aE = o.applyAsDouble(aE);
                    long rIndex = this == r ? aIndex : result.linearIndex(i, j);
                    result.buffer.setAtIndex(ValueLayout.JAVA_DOUBLE, rIndex, aE);
                }
            }
        }
    }

    // Binary, update a in place
    public void elementwise(Matrix<Double> b, DoubleBinaryOperator o) {
        elementwise(b, this, o);
    }

    // Binary, update to r
    public void elementwise(Matrix<Double> b, Matrix<Double> r, DoubleBinaryOperator o) {
        DoubleMatrix that = cast(b);
        DoubleMatrix result = cast(r);

        if (rows != result.rows || columns != result.columns) {
            throw new IllegalArgumentException("All matrices must have the same dimensions");
        }

        // Same rows and columns
        if (rows == that.rows && columns == that.columns) {
            elementwiseSameSize(this, that, result, o);
        }
        // B is a scalar
        else if (that.rows == 1 && that.columns == 1) {
            long bIndex = that.linearIndex(0, 0);
            double bE = that.buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, bIndex);

            elementwiseWithScalar(this, bE, result, o);
        }
        // Same columns
        // B is a row-vector
        else if (that.rows == 1 && columns == that.columns) {
            elementwiseRowVector(this, that, result, o);
        }
        // Same rows
        // B is a column-vector
        else if (rows == that.rows && that.columns == 1) {
            elementwiseColumnVector(this, that, result, o);
        } else {
            throw new IllegalArgumentException(
                    String.format("Mismatched matrices: a(%d, %d) x b(%d, %d)", rows, columns, that.rows, that.columns));
        }
    }


    /*
    A and B are the same size

     a a a a a   b b b b b
     a a a a a X b b b b b
     a a a a a   b b b b b

     Broadcast B over A
     */
    private static void elementwiseSameSize(DoubleMatrix a, DoubleMatrix b, DoubleMatrix r,
                                            DoubleBinaryOperator o) {
        assert a.rows == b.rows && a.columns == b.columns;

        if (a.isZeroOffset() && b.isZeroOffset() && r.isZeroOffset() &&
                o instanceof DoubleBroadcastKernel.BinaryBinaryBroadcastKernel k) {
            k.binaryBroadcastMatrix(a.buffer, b.buffer, r.buffer, a.rows * a.columns);
        }
        else {
            for (long j = 0; j < a.columns; j++) {
                for (long i = 0; i < a.rows; i++) {
                    long aIndex = a.linearIndex(i, j);
                    double aE = a.buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, aIndex);
                    long bIndex = b.linearIndex(i, j);
                    double bE = b.buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, bIndex);
                    aE = o.applyAsDouble(aE, bE);
                    long rIndex = a == r ? aIndex : r.linearIndex(i, j);
                    r.buffer.setAtIndex(ValueLayout.JAVA_DOUBLE, rIndex, aE);
                }
            }
        }
    }

    /*
    B is a scalar

    a a a a a   b
    a a a a a X
    a a a a a

    Broadcast b over A
     */
    private static void elementwiseWithScalar(DoubleMatrix a, double bE, DoubleMatrix r,
                                              DoubleBinaryOperator o) {
        if (a.isZeroOffset() && r.isZeroOffset() &&
                o instanceof DoubleBroadcastKernel.BinaryBinaryBroadcastKernel k) {
            k.binaryBroadcastScalar(a.buffer, bE, r.buffer, 0, a.rows * a.columns);
        } else {
            for (long j = 0; j < a.columns; j++) {
                for (long i = 0; i < a.rows; i++) {
                    long aIndex = a.linearIndex(i, j);
                    double aE = a.buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, aIndex);
                    aE = o.applyAsDouble(aE, bE);
                    long rIndex = a == r ? aIndex : r.linearIndex(i, j);
                    r.buffer.setAtIndex(ValueLayout.JAVA_DOUBLE, rIndex, aE);
                }
            }
        }
    }

    /*
    Same columns
    B is a row-vector

    a a a a a   b b b b b
    a a a a a X
    a a a a a

    Broadcast B over the rows of A
     */
    private static void elementwiseRowVector(DoubleMatrix a, DoubleMatrix b, DoubleMatrix r,
                                             DoubleBinaryOperator o) {
        assert b.rows == 1;

        if (a.isZeroOffset() && b.isZeroOffset() && r.isZeroOffset() &&
                a.rowStride == 1 && r.rowStride == 1 &&
                b.columnStride == 1 &&
                o instanceof DoubleBroadcastKernel.BinaryBinaryBroadcastKernel k) {
            k.binaryBroadcastVector(a.buffer, b.buffer, r.buffer, a.columnStride, b.columns);
        }
        else {
            for (long j = 0; j < a.columns; j++) {
                long bIndex = b.linearIndex(0, j);
                double bE = b.buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, bIndex);
                // Apply b to row of A

                for (long i = 0; i < a.rows; i++) {
                    long aIndex = a.linearIndex(i, j);
                    double aE = a.buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, aIndex);
                    aE = o.applyAsDouble(aE, bE);
                    long rIndex = a == r ? aIndex : r.linearIndex(i, j);
                    r.buffer.setAtIndex(ValueLayout.JAVA_DOUBLE, rIndex, aE);
                }
            }
        }
    }

    /*
    Same rows
    B is a column-vector

    a a a a a   b
    a a a a a X b
    a a a a a   b

    Broadcast B over the columns of A
     */
    private static void elementwiseColumnVector(DoubleMatrix a, DoubleMatrix b, DoubleMatrix r,
                                                DoubleBinaryOperator o) {
        assert b.columns == 1;

        if (a.isZeroOffset() && b.isZeroOffset() && r.isZeroOffset() &&
                a.columnStride == 1 && r.columnStride == 1 &&
                b.rowStride == 1 &&
                o instanceof DoubleBroadcastKernel.BinaryBinaryBroadcastKernel k) {
            assert a.rowStride == b.rows;
            k.binaryBroadcastVector(a.buffer, b.buffer, r.buffer, a.rowStride, b.rows);
        }
        else {
            for (long i = 0; i < a.rows; i++) {
                long bIndex = b.linearIndex(i, 0);
                double bE = b.buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, bIndex);
                // Apply b to column of A

                for (long j = 0; j < a.columns; j++) {
                    long aIndex = a.linearIndex(i, j);
                    double aE = a.buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, aIndex);
                    aE = o.applyAsDouble(aE, bE);
                    long rIndex = a == r ? aIndex : r.linearIndex(i, j);
                    r.buffer.setAtIndex(ValueLayout.JAVA_DOUBLE, rIndex, aE);
                }
            }
        }
    }


    // Binary, update a in place
    public void elementwise(Matrix<Double> b, Matrix<Double> c, DoubleTernaryOperator o) {
        elementwise(b, c, this, o);
    }

    // Binary, update to r
    public void elementwise(Matrix<Double> b, Matrix<Double> c, Matrix<Double> r, DoubleTernaryOperator o) {
        DoubleMatrix bM = cast(b);
        DoubleMatrix cM = cast(c);
        DoubleMatrix result = cast(r);

        if (rows != result.rows || columns != result.columns || bM.rows != cM.rows || bM.columns != cM.columns) {
            throw new IllegalArgumentException("All matrices must have the same dimensions");
        }

        // Same rows and columns
        if (rows == bM.rows && columns == bM.columns) {
            elementwiseSameSize(this, bM, cM, result, o);
        }
        // B is a scalar
        else if (bM.rows == 1 && bM.columns == 1) {
            long bIndex = bM.linearIndex(0, 0);
            double bE = bM.buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, bIndex);
            double cE = cM.buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, bIndex);

            elementwiseWithScalar(this, bE, cE, result, o);
        }
        // Same columns
        // B is a row-vector
        else if (bM.rows == 1 && columns == bM.columns) {
            elementwiseRowVector(this, bM, cM, result, o);
        }
        // Same rows
        // B is a column-vector
        else if (rows == bM.rows && bM.columns == 1) {
            elementwiseColumnVector(this, bM, cM, result, o);
        } else {
            throw new IllegalArgumentException(
                    String.format("Mismatched matrices: a(%d, %d) x b(%d, %d)", rows, columns, bM.rows, bM.columns));
        }
    }

    /*
    A and B are the same size

     a a a a a   b b b b b
     a a a a a X b b b b b
     a a a a a   b b b b b

     Broadcast B over A
     */
    private static void elementwiseSameSize(DoubleMatrix a, DoubleMatrix b, DoubleMatrix c, DoubleMatrix r,
                                            DoubleTernaryOperator o) {
        assert a.rows == b.rows && a.columns == b.columns;

        if (a.isZeroOffset() && b.isZeroOffset() && c.isZeroOffset() && r.isZeroOffset() &&
                o instanceof DoubleBroadcastKernel.TernaryBroadcastKernel k) {
            k.binaryBroadcastMatrix(a.buffer, b.buffer, c.buffer, r.buffer, a.rows * a.columns);
        }
        else {
            for (long j = 0; j < a.columns; j++) {
                for (long i = 0; i < a.rows; i++) {
                    long aIndex = a.linearIndex(i, j);
                    double aE = a.buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, aIndex);
                    long bIndex = b.linearIndex(i, j);
                    double bE = b.buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, bIndex);
                    long cIndex = c.linearIndex(i, j);
                    double cE = c.buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, cIndex);
                    aE = o.applyAsDouble(aE, bE, cE);
                    long rIndex = a == r ? aIndex : r.linearIndex(i, j);
                    r.buffer.setAtIndex(ValueLayout.JAVA_DOUBLE, rIndex, aE);
                }
            }
        }
    }

    /*
    B is a scalar

    a a a a a   b
    a a a a a X
    a a a a a

    Broadcast b over A
     */
    private static void elementwiseWithScalar(DoubleMatrix a, double bE, double cE, DoubleMatrix r,
                                              DoubleTernaryOperator o) {
        if (a.isZeroOffset() && r.isZeroOffset() &&
                o instanceof DoubleBroadcastKernel.TernaryBroadcastKernel k) {
            k.binaryBroadcastScalar(a.buffer, bE, cE, r.buffer, 0, a.rows * a.columns);
        } else {
            for (long j = 0; j < a.columns; j++) {
                for (long i = 0; i < a.rows; i++) {
                    long aIndex = a.linearIndex(i, j);
                    double aE = a.buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, aIndex);
                    aE = o.applyAsDouble(aE, bE, cE);
                    long rIndex = a == r ? aIndex : r.linearIndex(i, j);
                    r.buffer.setAtIndex(ValueLayout.JAVA_DOUBLE, rIndex, aE);
                }
            }
        }
    }

    /*
    Same columns
    B is a row-vector

    a a a a a   b b b b b
    a a a a a X
    a a a a a

    Broadcast B over the rows of A
     */
    private static void elementwiseRowVector(DoubleMatrix a, DoubleMatrix b, DoubleMatrix c, DoubleMatrix r,
                                             DoubleTernaryOperator o) {
        assert b.rows == 1;

        if (a.isZeroOffset() && b.isZeroOffset() && c.isZeroOffset() && r.isZeroOffset() &&
                a.rowStride == 1 && r.rowStride == 1 &&
                b.columnStride == 1 &&
                o instanceof DoubleBroadcastKernel.TernaryBroadcastKernel k) {
            k.binaryBroadcastVector(a.buffer, b.buffer, c.buffer, r.buffer, a.columnStride, b.columns);
        }
        else {
            for (long j = 0; j < a.columns; j++) {
                long bIndex = b.linearIndex(0, j);
                double bE = b.buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, bIndex);
                long cIndex = c.linearIndex(0, j);
                double cE = c.buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, cIndex);
                // Apply b to row of A

                for (long i = 0; i < a.rows; i++) {
                    long aIndex = a.linearIndex(i, j);
                    double aE = a.buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, aIndex);
                    aE = o.applyAsDouble(aE, bE, cE);
                    long rIndex = a == r ? aIndex : r.linearIndex(i, j);
                    r.buffer.setAtIndex(ValueLayout.JAVA_DOUBLE, rIndex, aE);
                }
            }
        }
    }

    /*
    Same rows
    B is a column-vector

    a a a a a   b
    a a a a a X b
    a a a a a   b

    Broadcast B over the columns of A
     */
    private static void elementwiseColumnVector(DoubleMatrix a, DoubleMatrix b, DoubleMatrix c, DoubleMatrix r,
                                                DoubleTernaryOperator o) {
        assert b.columns == 1;

        if (a.isZeroOffset() && b.isZeroOffset() && c.isZeroOffset() && r.isZeroOffset() &&
                a.columnStride == 1 && r.columnStride == 1 &&
                b.rowStride == 1 &&
                o instanceof DoubleBroadcastKernel.TernaryBroadcastKernel k) {
            assert a.rowStride == b.rows;
            k.binaryBroadcastVector(a.buffer, b.buffer, c.buffer, r.buffer, a.rowStride, b.rows);
        }
        else {
            for (long i = 0; i < a.rows; i++) {
                long bIndex = b.linearIndex(i, 0);
                double bE = b.buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, bIndex);
                long cIndex = c.linearIndex(i, 0);
                double cE = c.buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, cIndex);
                // Apply b to column of A

                for (long j = 0; j < a.columns; j++) {
                    long aIndex = a.linearIndex(i, j);
                    double aE = a.buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, aIndex);
                    aE = o.applyAsDouble(aE, bE, cE);
                    long rIndex = a == r ? aIndex : r.linearIndex(i, j);
                    r.buffer.setAtIndex(ValueLayout.JAVA_DOUBLE, rIndex, aE);
                }
            }
        }
    }


    // statistics -> 4 columns of min, max, mean, medium, rows is input columns

    public void reductionRow(Matrix<Double> r, DoubleBinaryOperator o) {
        DoubleMatrix result = cast(r);

        if (result.columns != 1 || result.rows != rows) {
            throw new IllegalArgumentException();
        }

        if (isZeroOffset() && r.isZeroOffset() &&
                columnStride == 1 &&
                o instanceof DoubleBroadcastKernel.ReductionBroadcastKernel k) {
            k.reduceBroadcastVector(buffer, r.buffer, rowStride, result.rows);
        }
        else {
            for (long i = 0; i < rows; i++) {
                double acc = 0.0;
                for (long j = 0; j < columns; j++) {
                    long index = linearIndex(i, j);
                    double e = buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, index);
                    acc = o.applyAsDouble(acc, e);
                }
                result.set(i, 0, acc);
            }
        }
    }

    public void reductionColumn(Matrix<Double> r, DoubleBinaryOperator o) {
        DoubleMatrix result = cast(r);

        if (result.rows != 1 || result.columns != columns) {
            throw new IllegalArgumentException();
        }

        if (isZeroOffset() && r.isZeroOffset() &&
                rowStride == 1 &&
                o instanceof DoubleBroadcastKernel.ReductionBroadcastKernel k) {
            k.reduceBroadcastVector(buffer, r.buffer, columnStride, result.columns);
        }
        else {
            for (long j = 0; j < columns; j++) {
                double acc = 0.0;
                for (long i = 0; i < rows; i++) {
                    long index = linearIndex(i, j);
                    double e = buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, index);
                    acc = o.applyAsDouble(acc, e);
                }
                result.set(0, j, acc);
            }
        }
    }

    @FunctionalInterface
    public interface ReductionSupplier<T> {
        T apply(long j);
    }

    @FunctionalInterface
    public interface ReductionAccumulator<T> {
        T apply(T t, double v);
    }

    @FunctionalInterface
    public interface ReductionFinisher<T> {
        double apply(T t);
    }

    public <T> void reductionColumn(Matrix<Double> r,
                                    ReductionSupplier<T> s, ReductionAccumulator<T> a, ReductionFinisher<T> f) {
        DoubleMatrix result = cast(r);

        if (result.rows != 1 || result.columns != columns) {
            throw new IllegalArgumentException();
        }

        for (long j = 0; j < columns; j++) {
            T acc = s.apply(j);
            for (long i = 0; i < rows; i++) {
                long index = linearIndex(i, j);
                double e = buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, index);
                acc = a.apply(acc, e);
            }
            result.set(0, j, f.apply(acc));
        }
    }

    public double reduction(DoubleBinaryOperator o) {
        if (isZeroOffset()
                && o instanceof DoubleBroadcastKernel.ReductionBroadcastKernel k) {
            return k.reduceBroadcastScalar(buffer, 0, rows * columns);
        }
        else {
            double acc = 0.0;
            for (long j = 0; j < columns; j++) {
                for (long i = 0; i < rows; i++) {
                    long index = linearIndex(i, j);
                    double e = buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, index);
                    acc = o.applyAsDouble(acc, e);
                }
            }
            return acc;
        }
    }

    @FunctionalInterface
    public interface MatrixConsumer {
        void accept(long i, long j, double v);
    }

    @FunctionalInterface
    public interface MatrixSupplier {
        double accept(long i, long j, double v);
    }

    public void forEach(MatrixConsumer c) {
        for (long j = 0; j < columns; j++) {
            for (long i = 0; i < rows; i++) {
                long index = linearIndex(i, j);
                double e = buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, index);
                c.accept(i, j, e);
            }
        }
    }

    public void setEach(MatrixSupplier c) {
        for (long j = 0; j < columns; j++) {
            for (long i = 0; i < rows; i++) {
                long index = linearIndex(i, j);
                double e = buffer.getAtIndex(ValueLayout.JAVA_DOUBLE, index);
                e = c.accept(i, j, e);
                buffer.setAtIndex(ValueLayout.JAVA_DOUBLE, index, e);
            }
        }
    }

    @Override
    public DoubleMatrix subMatrix(long i, long j, long rows, long columns) {
        if (i + rows > this.rows || j + columns > this.columns) {
            throw new IllegalArgumentException();
        }

        MemorySegment view = obj_t.allocate(allocator);
        blis_h.bli_acquire_mpart(i, j, rows, columns, this.obj, view);
        return new DoubleMatrix(allocator, view, buffer);
    }

    @Override
    public DoubleMatrix withConjTrans(Trans t) {
        return (DoubleMatrix) super.withConjTrans(t);
    }

    @Override
    public DoubleMatrix withTransOnly(Trans t) {
        return (DoubleMatrix) super.withTransOnly(t);
    }

    @Override
    public void copyInto(Matrix<Double> r) {
        if (rows != r.rows || columns != r.columns) {
            throw new IllegalArgumentException();
        }

        // @@@ TODO implement in Java?
        blis_h.bli_copym(obj, cast(r).obj);
    }

    public void extractColumnInto(int column, double[] r, int offset) {
        Objects.checkIndex(column, columns);
        Objects.checkIndex(offset, r.length);
        Objects.checkFromIndexSize(offset, rows, r.length);

        for (int i = 0; i < rows; i++) {
            r[offset + i] = get(i, column);
        }
    }

    @Override
    public void transpose(Matrix<Double> r) {
        DoubleMatrix result = cast(r);

        if (columns != r.rows || rows != r.columns) {
            throw new IllegalArgumentException();
        }

        for (int j = 0; j < columns; j++) {
            for (int i = 0; i < rows; i++) {
                result.set(j, i, get(i, j));
            }
        }
    }

    @Override
    public DoubleMatrix transpose() {
        return transpose(allocator);
    }

    @Override
    public DoubleMatrix transpose(SegmentAllocator sa) {
        DoubleMatrix t = newDoubleMatrix(sa, columns, rows);
        transpose(t);
        return t;
    }

    // a a a
    // a a a
    // a a a
    //    x
    // b b b
    // b b b
    // b b b
    public void concatVertically(Matrix<Double> b, Matrix<Double> r) {
        if (b.columns != columns || r.columns != columns || r.rows < b.rows + rows) {
            throw new IllegalArgumentException();
        }
        cast(b);
        cast(r);

        copyInto(r.subMatrix(0, 0, rows, columns));
        b.copyInto(r.subMatrix(rows, 0, b.rows, columns));
    }

    // a a a    b b b
    // a a a  x b b b
    // a a a    b b b
    public void concatHorizontally(Matrix<Double> b, Matrix<Double> c) {
        if (b.rows != rows || c.rows != rows || c.columns < b.columns + columns) {
            throw new IllegalArgumentException();
        }
        cast(b);
        cast(c);

        copyInto(c.subMatrix(0, 0, rows, columns));
        b.copyInto(c.subMatrix(0, columns, rows, b.columns));
    }

    private static DoubleMatrix cast(Matrix<Double> dest) {
        return (DoubleMatrix) dest;
    }

    // Apply Double.sum(a, b) to all elements
    public void add(Matrix<Double> b) {
        elementwise(b, Double::sum);
    }

    // Apply Double.sum(acc, a) to all elements
    public double sum() {
        return reduction(Double::sum);
    }

    public double mean() {
        return sum() / (rows * columns);
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof DoubleMatrix that)) {
            return false;
        }

        if (rows != that.rows || columns != that.columns) {
            return false;
        }

        for (int j = 0; j < columns; j++) {
            for (int i = 0; i < rows; i++) {
                double e = get(i, j);
                double thatE = that.get(i, j);

                if (e != thatE) {
                    return false;
                }
            }
        }

        return true;
    }

    public boolean equals(Matrix<Double> m, double epsilon) {
        DoubleMatrix that = cast(m);

        if (rows != that.rows || columns != that.columns) {
            return false;
        }

        for (int j = 0; j < columns; j++) {
            for (int i = 0; i < rows; i++) {
                double e = get(i, j);
                double thatE = that.get(i, j);

                if (Math.abs(e - thatE) > epsilon) {
                    return false;
                }
            }
        }

        return true;
    }
}
