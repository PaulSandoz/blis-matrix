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

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.MemorySession;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.ValueLayout;

public abstract sealed class Matrix<T> permits DoubleMatrix, Matrix.PolymorphicConstant {
    // Allocator to allocate view obj_t
    final SegmentAllocator allocator;

    // Pointer to BLIS object of obj_t
    final MemorySegment obj;

    // Pointer to buffer of elements
    final MemorySegment buffer;

    // Size information (cloned from obj_t)
    final long rows;   // length, number of rows
    final long columns;   // width, number of columns
    final long rowOffset;  // row offset
    final long columnOffset;  // column offset
    final long rowStride;  // row stride
    final long columnStride;  // column stride


    Matrix(SegmentAllocator allocator, MemorySegment obj, MemorySegment buffer) {
        this.allocator = allocator;
        this.obj = obj;
        this.buffer = buffer;
        MemorySegment dim = obj_t.dim$slice(obj);
        this.rows = dim.getAtIndex(ValueLayout.JAVA_LONG, 0);
        this.columns = dim.getAtIndex(ValueLayout.JAVA_LONG, 1);
        MemorySegment off = obj_t.off$slice(obj);
        this.rowOffset = off.getAtIndex(ValueLayout.JAVA_LONG, 0);
        this.columnOffset = off.getAtIndex(ValueLayout.JAVA_LONG, 1);
        this.rowStride = obj_t.rs$get(obj);
        this.columnStride = obj_t.cs$get(obj);
    }

    public String toDimString() {
        return String.format("[%d, %d]", rows, columns);
    }

    final long linearIndex(long i, long j) {
        return (rowOffset + i) * rowStride + (columnOffset + j) * columnStride;
    }

    final boolean isZeroOffset() {
        return rowOffset == 0 && columnOffset == 0;
    }

    public MemorySegment buffer() {
        return buffer;
    }

    public long rows() {
        return rows;
    }

    public long columns() {
        return columns;
    }

    public long rowStride() {
        return rowStride;
    }

    public long columnStride() {
        return columnStride;
    }

    public long rowOffset() {
        return rowOffset;
    }

    public long columnOffset() {
        return columnOffset;
    }

    public void randm() {
        blis_h.bli_randm(obj);
    }

    public abstract Matrix<T> subMatrix(long i, long j, long rows, long columns);

    public abstract void copyInto(Matrix<T> dest);

    public abstract void transpose(Matrix<T> b);

    public abstract Matrix<T> transpose();

    public abstract Matrix<T> transpose(SegmentAllocator sa);

    public abstract void concatVertically(Matrix<T> b, Matrix<T> c);

    public abstract void concatHorizontally(Matrix<T> b, Matrix<T> c);

    public void print() {
        print("");
    }

    public void print(String m) {
        try (var scope = MemorySession.openConfined()) {
            var message = scope.allocateUtf8String(m);
            var emptyString = scope.allocateUtf8String("");
            var format = scope.allocateUtf8String("%5.2f");
            blis_h.bli_printm(message, obj, format, emptyString);
        }
    }

    // Properties

    public enum Trans {
        NO_TRANSPOSE(blis_h.BLIS_NO_TRANSPOSE()),
        TRANSPOSE(blis_h.BLIS_TRANSPOSE()),
        CONJ_NO_TRANSPOSE(blis_h.BLIS_CONJ_NO_TRANSPOSE()),
        CONJ_TRANSPOSE(blis_h.BLIS_CONJ_TRANSPOSE()),
        ;

        final int v;

        Trans(int v) {
            this.v = v;
        }

        static Trans from(int v) {
            for (var e : Trans.values()) {
                if (e.v == v) {
                    return e;
                }
            }
            throw new IllegalArgumentException("No struc for value " + v);
        }
    }

    public enum Conj {
        NO_CONJUGATE(blis_h.BLIS_NO_CONJUGATE()),
        CONJUGATE(blis_h.BLIS_CONJUGATE()),
        ;

        final int v;

        Conj(int v) {
            this.v = v;
        }

        static Conj from(int v) {
            for (var e : Conj.values()) {
                if (e.v == v) {
                    return e;
                }
            }
            throw new IllegalArgumentException("No struc for value " + v);
        }
    }

    public enum Structure {
        GENERAL(blis_h.BLIS_GENERAL()),
        HERMITIAN(blis_h.BLIS_HERMITIAN()),
        SYMMETRIC(blis_h.BLIS_SYMMETRIC()),
        TRIANGULAR(blis_h.BLIS_TRIANGULAR()),
        ;

        final int v;

        Structure(int v) {
            this.v = v;
        }

        static Structure from(int v) {
            for (var e : Structure.values()) {
                if (e.v == v) {
                    return e;
                }
            }
            throw new IllegalArgumentException("No struc for value " + v);
        }
    }

    public enum Uplo {
        ZEROS(blis_h.BLIS_ZEROS()),
        LOWER(blis_h.BLIS_LOWER()),
        UPPER(blis_h.BLIS_UPPER()),
        DENSE(blis_h.BLIS_DENSE()),
        ;

        final int v;

        Uplo(int v) {
            this.v = v;
        }

        static Uplo from(int v) {
            for (var e : Uplo.values()) {
                if (e.v == v) {
                    return e;
                }
            }
            throw new IllegalArgumentException("No struc for value " + v);
        }
    }

    public enum Diag {
        NONUNIT(blis_h.BLIS_NONUNIT_DIAG()),
        UNIT(blis_h.BLIS_UNIT_DIAG()),
        ;

        final int v;

        Diag(int v) {
            this.v = v;
        }

        static Diag from(int v) {
            for (var e : Diag.values()) {
                if (e.v == v) {
                    return e;
                }
            }
            throw new IllegalArgumentException("No struc for value " + v);
        }
    }

    public enum Side {
        LEFT(blis_h.BLIS_LEFT()),
        RIGHT(blis_h.BLIS_RIGHT()),
        ;

        final int v;

        Side(int v) {
            this.v = v;
        }

        static Side from(int v) {
            for (var e : Side.values()) {
                if (e.v == v) {
                    return e;
                }
            }
            throw new IllegalArgumentException("No struc for value " + v);
        }
    }

    public Trans conjTrans() {
        int info = obj_t.info$get(obj);
        return Trans.from(info & blis_h.BLIS_CONJTRANS_BITS());
    }

    public Matrix<T> withConjTrans(Trans t) {
        int i = obj_t.info$get(obj);
        i = (i & ~blis_h.BLIS_CONJTRANS_BITS()) | t.v;

        Matrix<T> that = subMatrix(0, 0, rows(), columns());
        obj_t.info$set(that.obj, i);
        return that;
    }

    public Trans transOnly() {
        int i = obj_t.info$get(obj);
        return Trans.from(i & blis_h.BLIS_TRANS_BIT());
    }

    public Matrix<T> withTransOnly(Trans t) {
        int i = obj_t.info$get(obj);
        i = (i & ~blis_h.BLIS_TRANS_BIT()) | t.v;

        Matrix<T> that = subMatrix(0, 0, rows(), columns());
        obj_t.info$set(that.obj, i);
        return that;
    }

    public Structure struc() {
        int info = obj_t.info$get(obj);
        return Structure.from(info & blis_h.BLIS_STRUC_BITS());
    }

    public Matrix<T> withStruc(Structure t) {
        int i = obj_t.info$get(obj);
        i = (i & ~blis_h.BLIS_STRUC_BITS()) | t.v;

        Matrix<T> that = subMatrix(0, 0, rows(), columns());
        obj_t.info$set(that.obj, i);
        return that;
    }

    public Uplo uplo() {
        int info = obj_t.info$get(obj);
        return Uplo.from(info & blis_h.BLIS_UPLO_BITS());
    }

    public Matrix<T> withUplo(Uplo t) {
        int i = obj_t.info$get(obj);
        i = (i & ~blis_h.BLIS_UPLO_BITS()) | t.v;

        Matrix<T> that = subMatrix(0, 0, rows(), columns());
        obj_t.info$set(that.obj, i);
        return that;
    }

    public Diag diag() {
        int i = obj_t.info$get(obj);
        return Diag.from(i & blis_h.BLIS_UNIT_DIAG_BIT());
    }

    public Matrix<T> withDiag(Diag d) {
        int i = obj_t.info$get(obj);
        i = (i & ~blis_h.BLIS_UNIT_DIAG_BIT()) | d.v;

        Matrix<T> that = subMatrix(0, 0, rows(), columns());
        obj_t.info$set(that.obj, i);
        return that;
    }

    public long diagOffset() {
        return obj_t.diag_off$get(obj);
    }

    public void setDiagOffset(long o) {
        obj_t.diag_off$set(obj, o);
    }

    // Specializations

    static final class PolymorphicConstant extends Matrix<Object> {
        PolymorphicConstant(MemorySegment obj) {
            super(null, obj, null);
        }

        @Override
        public Matrix<Object> subMatrix(long i, long j, long rows, long columns) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void copyInto(Matrix<Object> dest) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void transpose(Matrix<Object> b) {
            throw new UnsupportedOperationException();
        }

        public Matrix<Object> transpose() {
            throw new UnsupportedOperationException();

        }

        public Matrix<Object> transpose(SegmentAllocator sa) {
            throw new UnsupportedOperationException();
        }

        public void concatVertically(Matrix<Object> b, Matrix<Object> c) {
            throw new UnsupportedOperationException();
        }

        public void concatHorizontally(Matrix<Object> b, Matrix<Object> c) {
            throw new UnsupportedOperationException();
        }
    }

    private static final class Constants {
        static final Matrix<?> ZERO = new Matrix.PolymorphicConstant(blis_h.BLIS_ZERO$SEGMENT());
        static final Matrix<?> ONE = new Matrix.PolymorphicConstant(blis_h.BLIS_ONE$SEGMENT());
        static final Matrix<?> TWO = new Matrix.PolymorphicConstant(blis_h.BLIS_TWO$SEGMENT());
    }

    public static Matrix<?> zero() {
        return Constants.ZERO;
    }

    public static Matrix<?> one() {
        return Constants.ONE;
    }

    public static Matrix<?> two() {
        return Constants.TWO;
    }

    public static DoubleMatrix newDoubleMatrix(long rows, long columns) {
        return newDoubleMatrix(MemorySession.openImplicit(), rows, columns);
    }

    public static DoubleMatrix newDoubleMatrix(MemorySession scope, long rows, long columns) {
        return newDoubleMatrix((SegmentAllocator) scope, rows, columns);
    }

    static DoubleMatrix newDoubleMatrix(SegmentAllocator allocator, long rows, long columns) {
        // Allocate the memory for the matrix elements
        MemorySegment buffer = allocator.allocate(MemoryLayout.sequenceLayout(rows * columns, ValueLayout.JAVA_DOUBLE));
        return newDoubleMatrix(allocator, rows, columns, buffer);
    }

    static DoubleMatrix newDoubleMatrix(SegmentAllocator allocator, long rows, long columns, MemorySegment buffer) {
        MemorySegment obj = newObj_t(allocator, rows, columns, buffer);
        return new DoubleMatrix(allocator, obj, buffer);
    }

    static MemorySegment newObj_t(SegmentAllocator allocator, long rows, long columns, MemorySegment buffer) {
        // Allocate the obj_t struct and attach the buffer
        MemorySegment obj = obj_t.allocate(allocator);
        blis_h.bli_obj_create_with_attached_buffer(
                // Element type
                blis_h.BLIS_DOUBLE(),
                // Shape
                rows, columns,
                // Pointer to elements
                buffer,
                // Row and column strides
                // Column-major order
                1, rows,
                obj);
        return obj;
    }
}