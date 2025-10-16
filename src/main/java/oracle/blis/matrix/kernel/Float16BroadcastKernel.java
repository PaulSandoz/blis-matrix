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

package oracle.blis.matrix.kernel;

import jdk.incubator.vector.Float16;
import oracle.blis.matrix.f.TernaryOperator;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;
import java.util.stream.IntStream;

/**
 * Kernels for unary, binary, ternary, and reduction operations that operate over contiguous regions of memory.
 */
public interface Float16BroadcastKernel {

    interface UnaryBroadcastKernel extends UnaryOperator<Float16> {
        default void unaryBroadcast(MemorySegment a, MemorySegment r,
                                    long length) {
            for (long i = 0; i < length; i++) {
                Float16 aE = Float16.shortBitsToFloat16(a.getAtIndex(ValueLayout.JAVA_SHORT, i));
                Float16 rE = apply(aE);
                r.setAtIndex(ValueLayout.JAVA_SHORT, i, Float16.float16ToRawShortBits(rE));
            }
        }
    }

    abstract class ParallelUnaryBroadcastKernel implements UnaryBroadcastKernel {
        final boolean parallel;

        public ParallelUnaryBroadcastKernel() {
            this(false);
        }

        public ParallelUnaryBroadcastKernel(boolean parallel) {
            this.parallel = parallel;
        }

        // @@@ parallel unaryBroadcast
    }

    interface BinaryBroadcastKernel extends BinaryOperator<Float16> {
        default void binaryBroadcastScalar(MemorySegment a, Float16 b, MemorySegment r,
                                           long offset, long length) {
            for (long i = 0; i < length; i++) {
                Float16 aE = Float16.shortBitsToFloat16(a.getAtIndex(ValueLayout.JAVA_SHORT, offset + i));
                Float16 rE = apply(aE, b);
                r.setAtIndex(ValueLayout.JAVA_SHORT, offset + i, Float16.float16ToRawShortBits(rE));
            }
        }

        default void binaryBroadcastVector(MemorySegment a, MemorySegment b, MemorySegment r,
                                           long aStride, long bLength) {
            for (long j = 0; j < bLength; j++) {
                Float16 bE = Float16.shortBitsToFloat16(b.getAtIndex(ValueLayout.JAVA_SHORT, j));
                binaryBroadcastScalar(a, bE, r, j * aStride, aStride);
            }
        }

        default void binaryBroadcastMatrix(MemorySegment a, MemorySegment b, MemorySegment r,
                                           long length) {
            for (long i = 0; i < length; i++) {
                Float16 aE = Float16.shortBitsToFloat16(a.getAtIndex(ValueLayout.JAVA_SHORT, i));
                Float16 bE = Float16.shortBitsToFloat16(b.getAtIndex(ValueLayout.JAVA_SHORT, i));
                Float16 rE = apply(aE, bE);
                r.setAtIndex(ValueLayout.JAVA_SHORT, i, Float16.float16ToRawShortBits(rE));
            }
        }
    }

    abstract class ParallelBinaryBroadcastKernel implements BinaryBroadcastKernel {
        final boolean parallel;

        public ParallelBinaryBroadcastKernel() {
            this(false);
        }

        public ParallelBinaryBroadcastKernel(boolean parallel) {
            this.parallel = parallel;
        }

        @Override
        public void binaryBroadcastVector(MemorySegment a, MemorySegment b, MemorySegment r, long aStride, long bLength) {
            if (parallel) {
                IntStream.range(0, (int) bLength).parallel().forEach(j -> {
                    Float16 bE = Float16.shortBitsToFloat16(b.getAtIndex(ValueLayout.JAVA_SHORT, j));
                    binaryBroadcastScalar(a, bE, r, j * aStride, aStride);
                });
            } else {
                BinaryBroadcastKernel.super.binaryBroadcastVector(a, b, r, aStride, bLength);
            }
        }

        // @@@ parallel binaryBroadcastMatrix
    }

    interface TernaryBroadcastKernel extends TernaryOperator<Float16> {
        default void binaryBroadcastScalar(MemorySegment a, Float16 b, Float16 c, MemorySegment r,
                                           long offset, long length) {
            for (long i = 0; i < length; i++) {
                Float16 aE = Float16.shortBitsToFloat16(a.getAtIndex(ValueLayout.JAVA_SHORT, offset + i));
                Float16 rE = apply(aE, b, c);
                r.setAtIndex(ValueLayout.JAVA_SHORT, offset + i, Float16.float16ToRawShortBits(rE));
            }
        }

        default void binaryBroadcastVector(MemorySegment a, MemorySegment b, MemorySegment c, MemorySegment r,
                                           long aStride, long bcLength) {
            for (long j = 0; j < bcLength; j++) {
                Float16 bE = Float16.shortBitsToFloat16(b.getAtIndex(ValueLayout.JAVA_SHORT, j));
                Float16 cE = Float16.shortBitsToFloat16(c.getAtIndex(ValueLayout.JAVA_SHORT, j));
                binaryBroadcastScalar(a, bE, cE, r, j * aStride, aStride);
            }
        }

        default void binaryBroadcastMatrix(MemorySegment a, MemorySegment b, MemorySegment c, MemorySegment r,
                                           long length) {
            for (long i = 0; i < length; i++) {
                Float16 aE = Float16.shortBitsToFloat16(a.getAtIndex(ValueLayout.JAVA_SHORT, i));
                Float16 bE = Float16.shortBitsToFloat16(b.getAtIndex(ValueLayout.JAVA_SHORT, i));
                Float16 cE = Float16.shortBitsToFloat16(c.getAtIndex(ValueLayout.JAVA_SHORT, i));
                Float16 rE = apply(aE, bE, cE);
                r.setAtIndex(ValueLayout.JAVA_SHORT, i, Float16.float16ToRawShortBits(rE));
            }
        }
    }

    abstract class ParallelTernaryBroadcastKernel implements TernaryBroadcastKernel {
        final boolean parallel;

        public ParallelTernaryBroadcastKernel() {
            this(false);
        }

        public ParallelTernaryBroadcastKernel(boolean parallel) {
            this.parallel = parallel;
        }

        @Override
        public void binaryBroadcastVector(MemorySegment a, MemorySegment b, MemorySegment c, MemorySegment r, long aStride, long bcLength) {
            if (parallel) {
                IntStream.range(0, (int) bcLength).parallel().forEach(j -> {
                    Float16 bE = Float16.shortBitsToFloat16(b.getAtIndex(ValueLayout.JAVA_SHORT, j));
                    Float16 cE = Float16.shortBitsToFloat16(c.getAtIndex(ValueLayout.JAVA_SHORT, j));
                    binaryBroadcastScalar(a, bE, cE, r, j * aStride, aStride);
                });
            } else {
                TernaryBroadcastKernel.super.binaryBroadcastVector(a, b, c, r, aStride, bcLength);
            }
        }

        // @@@ parallel binaryBroadcastMatrix
    }

    interface ReductionBroadcastKernel extends BinaryOperator<Float16> {
        default Float16 reduceBroadcastScalar(MemorySegment a,
                                             long offset, long length) {
            Float16 acc = Float16.valueOf(0.0);
            for (long i = 0; i < length; i++) {
                Float16 aE = Float16.shortBitsToFloat16(a.getAtIndex(ValueLayout.JAVA_SHORT, i + offset));
                acc = apply(acc, aE);
            }
            return acc;
        }

        default void reduceBroadcastVector(MemorySegment a, MemorySegment r,
                                           long aStride, long rLength) {
            for (long j = 0; j < rLength; j++) {
                Float16 rE = reduceBroadcastScalar(a, j * aStride, aStride);
                r.setAtIndex(ValueLayout.JAVA_SHORT, j, Float16.float16ToRawShortBits(rE));
            }
        }
    }

    abstract class ParallelReductionBroadcastKernel implements ReductionBroadcastKernel {
        final boolean parallel;

        public ParallelReductionBroadcastKernel() {
            this(false);
        }

        public ParallelReductionBroadcastKernel(boolean parallel) {
            this.parallel = parallel;
        }

        @Override
        public void reduceBroadcastVector(MemorySegment a, MemorySegment r, long aStride, long rLength) {
            if (parallel) {
                IntStream.range(0, (int) rLength).parallel().forEach(j -> {
                    Float16 rE = reduceBroadcastScalar(a, j * aStride, aStride);
                    r.setAtIndex(ValueLayout.JAVA_SHORT, j, Float16.float16ToRawShortBits(rE));
                });
            } else {
                Float16BroadcastKernel.ReductionBroadcastKernel.super.reduceBroadcastVector(a, r, aStride, rLength);
            }
        }
    }
}
