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

import oracle.blis.binding.obj_t;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static oracle.blis.binding.blis_h.*;

/*
On the Mac:

lldb -- \
    ~/Projects/jdk/jdk19/build/macosx-x86_64-server-release/images/jdk/bin/java \
    -cp target/classes:target/test-classes:bindings/target/blis/classes \
    -Djava.library.path=/Users/sandoz/Projects/jdk/blis/blis-flame/lib/haswell \
    --enable-native-access=ALL-UNNAMED \
    --enable-preview \
    -agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=*:5005 \
    oracle.blis.matrix.TestNativeDebug
 */
public class TestNativeDebug {
    public static void main(String[] args) {
        try (Arena sa = Arena.ofConfined()) {
            /* num_t  */ int dt;
            /* dim_t  */ long m, n, k;
            /* inc_t  */ long rs, cs;

            /* obj_t* */ MemorySegment a = obj_t.allocate(sa);
            /* obj_t* */ MemorySegment b = obj_t.allocate(sa);
            /* obj_t* */ MemorySegment c = obj_t.allocate(sa);
            /* obj_t* */ MemorySegment alpha;
            /* obj_t* */ MemorySegment beta;


            // Create some matrix operands to work with.
            dt = BLIS_DOUBLE();
            m = 4; n = 5; k = 3; rs = 0; cs = 0;
            bli_obj_create(dt, m, n, rs, cs, c);
            bli_obj_create(dt, m, k, rs, cs, a);
            bli_obj_create(dt, k, n, rs, cs, b);

            // Set the scalars to use.
            alpha = BLIS_ONE();
            beta = BLIS_ONE();

            // Initialize the matrix operands.
            bli_randm(a);
            bli_setm(BLIS_ONE(), b);
            bli_setm(BLIS_ZERO(), c);

            bli_printm(sa.allocateFrom("a: randomized"), a,
                    sa.allocateFrom("%5.2f"), sa.allocateFrom(""));
            bli_printm(sa.allocateFrom("b: set to 1.0"), b,
                    sa.allocateFrom("%5.2f"), sa.allocateFrom(""));
            bli_printm(sa.allocateFrom("c: initial value"), c,
                    sa.allocateFrom("%5.2f"), sa.allocateFrom(""));

            // c := beta * c + alpha * a * b, where 'a', 'b', and 'c' are general.
            System.out.printf("alpha=%#x, a=%#x,  b=%#x, beta=%#x, c=%#x\n",
                    alpha.address(),
                    a.address(),
                    b.address(),
                    beta.address(),
                    c.address());
            bli_gemm(alpha, a, b, beta, c);

            bli_printm(sa.allocateFrom("c: after gemm"), c,
                    sa.allocateFrom("%5.2f"), sa.allocateFrom(""));

            // Free the objects.
            bli_obj_free(a);
            bli_obj_free(b);
            bli_obj_free(c);
        }
    }

}
