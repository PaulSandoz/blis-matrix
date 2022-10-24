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

import org.junit.Test;

import java.lang.foreign.ValueLayout;

public class TestMatrix {

    static final BlisOperations BLI = BlisOperations.singleton();

    @Test
    public void testViews() {
        var m = DoubleMatrix.newDoubleMatrix(10, 4);
        m.randm();
        m.print();

        var sm = m.subMatrix(1, 1, 8, 2);
        sm.print();

        BLI.setm(Matrix.zero(), sm);

        m.print();
    }

    @Test
    public void testAccess() {
        var m = DoubleMatrix.newDoubleMatrix(10, 4);
        m.randm();
        m.print();

        System.out.println("(0, 0) = " +
                m.get(0, 0));
        System.out.println("(1, 0) = " +
                m.get(1, 0));
        System.out.println("(2, 2) = " +
                m.get(2, 2));

        System.out.println("[0] = " +
                m.buffer().getAtIndex(ValueLayout.JAVA_DOUBLE, 0));
        System.out.println("[1] = " +
                m.buffer().getAtIndex(ValueLayout.JAVA_DOUBLE, 1));
        System.out.println("[2 + 2 * 10] = " +
                m.buffer().getAtIndex(ValueLayout.JAVA_DOUBLE, 22));
    }

    @Test
    public void testElementalAccessUnary() {
        var m = DoubleMatrix.newDoubleMatrix(10, 4);
        m.setEach((i, j, v) -> j + i * m.columns());
        m.print();

        m.elementwise(v -> -v);
        m.print();
    }


    @Test
    public void testBinaryElementwise() {
        var m = DoubleMatrix.newDoubleMatrix(10, 4);
        m.setEach((i, j, v) -> j + i * m.columns());
        m.print();

        // Broadcast scalar
        var scalar = DoubleMatrix.newDoubleMatrix(1, 1);
        scalar.setEach((i, j, v) -> -1);
        m.elementwise(scalar, (a, b) -> a * b);
        m.print();

        // Broadcast vector
        var rowVector = DoubleMatrix.newDoubleMatrix(1, 4);
        rowVector.setEach((i, j, v) -> j % 2 == 0 ? 1 : -1);
        m.elementwise(rowVector, (a, b) -> a * b);
        m.print();

        // Broadcast column
        var columnVector = DoubleMatrix.newDoubleMatrix(10, 1);
        columnVector.setEach((i, j, v) -> i % 2 == 0 ? 1 : -1);
        m.elementwise(columnVector, (a, b) -> a * b);
        m.print();

        // Same dimensions
        var n = DoubleMatrix.newDoubleMatrix(10, 4);
        n.setEach((i, j, v) -> j + i * m.columns());
        m.elementwise(n, Double::sum);
        m.print();
    }

    @Test
    public void testColumnMeanAndStd() {
        var m = DoubleMatrix.newDoubleMatrix(10, 4);
        m.setEach((i, j, v) -> j + i * m.columns());
        m.print();

        var mean = Matrix.newDoubleMatrix(1, m.columns());
        // Column reduction producing a row vector
        m.reductionColumn(mean, Double::sum);
        mean.print();
        // Unary division
        mean.elementwise(e -> e / m.rows());
        mean.print();

        var std = columnStd(m, mean, true);
        std.print();

        // Scale by mean and std
        // Ternary elementwise
        m.elementwise(mean, std, (a, meanValue, stdValue) -> (a - meanValue) / stdValue);
        m.print();
    }

    static DoubleMatrix columnStd(DoubleMatrix data, DoubleMatrix mean, boolean isBiasCorrected) {
        DoubleMatrix std = Matrix.newDoubleMatrix(1, data.columns());
        double x = isBiasCorrected ? data.rows() - 1.0 : data.rows();
        data.reductionColumn(std,
                (j) -> {
                    var acc = new double[2];
                    acc[1] = mean.get(0, j);
                    return acc;
                },
                (acc, v) -> {
                    double d = v - acc[1];
                    acc[0] += d * d;
                    return acc;
                },
                acc -> Math.sqrt(acc[0] / x)
        );

        return std;
    }
}

