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

import oracle.blis.matrix.binding.obj_t;
import org.junit.Test;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.MemorySession;

import static oracle.blis.matrix.binding.blis_h.*;

public class TestApi {
    @Test
    public void testUsingBinding() {
        try (MemorySession s = MemorySession.openConfined()) {
            /* num_t  */ int dt;
            /* dim_t  */ long m, n, k;
            /* inc_t  */ long rs, cs;

            /* obj_t* */ MemorySegment a = obj_t.allocate(s);
            /* obj_t* */ MemorySegment b = obj_t.allocate(s);
            /* obj_t* */ MemorySegment c = obj_t.allocate(s);
            /* obj_t* */ MemorySegment alpha;
            /* obj_t* */ MemorySegment beta;


            // Create some matrix operands to work with.
            dt = BLIS_DOUBLE();
            m = 4; n = 5; k = 3; rs = 0; cs = 0;
            bli_obj_create(dt, m, n, rs, cs, c);
            bli_obj_create(dt, m, k, rs, cs, a);
            bli_obj_create(dt, k, n, rs, cs, b);

            // Set the scalars to use.
            alpha = BLIS_ONE$SEGMENT();
            beta = BLIS_ONE$SEGMENT();

            // Initialize the matrix operands.
            bli_randm(a);
            bli_setm(BLIS_ONE$SEGMENT(), b);
            bli_setm(BLIS_ZERO$SEGMENT(), c);

            bli_printm(s.allocateUtf8String("a: randomized"), a,
                    s.allocateUtf8String("%5.2f"), s.allocateUtf8String(""));
            bli_printm(s.allocateUtf8String("b: set to 1.0"), b,
                    s.allocateUtf8String("%5.2f"), s.allocateUtf8String(""));
            bli_printm(s.allocateUtf8String("c: initial value"), c,
                    s.allocateUtf8String("%5.2f"), s.allocateUtf8String(""));

            // c := beta * c + alpha * a * b, where 'a', 'b', and 'c' are general.
            bli_gemm(alpha, a, b, beta, c);

            bli_printm(s.allocateUtf8String("c: after gemm"), c,
                    s.allocateUtf8String("%5.2f"), s.allocateUtf8String(""));

            // Free the objects.
            bli_obj_free(a);
            bli_obj_free(b);
            bli_obj_free(c);
        }
    }

    static final BlisOperations BLI = BlisOperations.singleton();

    @Test
    public void testUsingMatrix() {
        try (MemorySession s = MemorySession.openConfined()) {
            /* dim_t  */ long m, n, k;

            // Create some matrix operands to work with.
            m = 4; n = 5; k = 3;
            DoubleMatrix c = Matrix.newDoubleMatrix(s, m, n);
            DoubleMatrix a = Matrix.newDoubleMatrix(s, m, k);
            DoubleMatrix b = Matrix.newDoubleMatrix(s, k, n);

            // Set the scalars to use.
            Matrix<?> alpha = Matrix.one();
            Matrix<?> beta = Matrix.one();

            // Initialize the matrix operands.
            BLI.randm(a);
            BLI.setm(DoubleMatrix.one(), b);
            // c's elements is already initialized to zero

            a.print("a: randomized");
            b.print("b: set to 1.0");
            c.print("b: set to 1.0");

            // c := beta * c + alpha * a * b, where 'a', 'b', and 'c' are general.
            BLI.gemm(alpha, a, b, beta, c);

            c.print("c: after gemm");
        }
    }

    @Test
    public void testUsingMatrixWithGC() {
        /* dim_t  */ long m, n, k;

        // Create some matrix operands to work with.
        m = 4; n = 5; k = 3;
        DoubleMatrix c = Matrix.newDoubleMatrix(m, n);
        DoubleMatrix a = Matrix.newDoubleMatrix(m, k);
        DoubleMatrix b = Matrix.newDoubleMatrix(k, n);

        // Set the scalars to use.
        Matrix<?> alpha = DoubleMatrix.one();
        Matrix<?> beta = DoubleMatrix.one();

        // Initialize the matrix operands.
        BLI.randm(a);
        BLI.setm(DoubleMatrix.one(), b);
        // c's elements is already initialized to zero

        a.print("a: randomized");
        b.print("b: set to 1.0");
        c.print("b: set to 1.0");

        // c := beta * c + alpha * a * b, where 'a', 'b', and 'c' are general.
        BLI.gemm(alpha, a, b, beta, c);

        c.print("c: after gemm");
    }
}
