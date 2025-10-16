package oracle.blis.matrix;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import oracle.blis.matrix.f.FloatBinaryOperator;
import oracle.blis.matrix.kernel.FloatBroadcastKernel;
import org.junit.jupiter.api.Test;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.function.BinaryOperator;

public class TestKernel {

    @Test
    public void testUsingScalar() {
        var m = FloatMatrix.newFloatMatrix(5, 3);
        m.setEach((i, j, v) -> j * m.rows() + i);
        m.print();

        var rowVector = FloatMatrix.newFloatMatrix(1, 3);
        rowVector.setEach((i, j, v) -> j);
        rowVector.print();

        m.elementwise(rowVector, (a, b) -> a * b);
        m.print();
    }

    static class PrintKernel implements FloatBroadcastKernel.BinaryBroadcastKernel {
        @Override
        public float applyAsFloat(float a, float b) {
            System.out.println("    applyAsFloat: " + a + " * " + b);
            return a * b;
        }

        @Override
        public void binaryBroadcastScalar(MemorySegment a, float b, MemorySegment r, long offset, long length) {
            System.out.println("  binaryBroadcastScalar: b = " + b + " offset = " + offset + " length = " + length);
            FloatBroadcastKernel.BinaryBroadcastKernel.super.binaryBroadcastScalar(a, b, r, offset, length);
        }

        @Override
        public void binaryBroadcastVector(MemorySegment a, MemorySegment b, MemorySegment r, long aStride, long bLength) {
            System.out.println("binaryBroadcastVector: aStride = " + aStride + " bLength = " + bLength);
            FloatBroadcastKernel.BinaryBroadcastKernel.super.binaryBroadcastVector(a, b, r, aStride, bLength);
        }
    }

    @Test
    public void testPrintKernel() {
        var m = FloatMatrix.newFloatMatrix(5, 3);
        m.setEach((i, j, v) -> j * m.rows() + i);
        m.print();

        var rowVector = FloatMatrix.newFloatMatrix(1, 3);
        rowVector.setEach((i, j, v) -> j);
        rowVector.print();

        m.elementwise(rowVector, new PrintKernel());
        m.print();
    }
/*
binaryBroadcastVector: aStride = 5 bLength = 3
  binaryBroadcastScalar: b = 0.0 offset = 0 length = 5
    applyAsFloat: 0.0 * 0.0
    applyAsFloat: 1.0 * 0.0
    applyAsFloat: 2.0 * 0.0
    applyAsFloat: 3.0 * 0.0
    applyAsFloat: 4.0 * 0.0
  binaryBroadcastScalar: b = 1.0 offset = 5 length = 5
    applyAsFloat: 5.0 * 1.0
    applyAsFloat: 6.0 * 1.0
    applyAsFloat: 7.0 * 1.0
    applyAsFloat: 8.0 * 1.0
    applyAsFloat: 9.0 * 1.0
  binaryBroadcastScalar: b = 2.0 offset = 10 length = 5
    applyAsFloat: 10.0 * 2.0
    applyAsFloat: 11.0 * 2.0
    applyAsFloat: 12.0 * 2.0
    applyAsFloat: 13.0 * 2.0
    applyAsFloat: 14.0 * 2.0
 */


    static class VectorMulKernel implements FloatBroadcastKernel.BinaryBroadcastKernel {
        static final VectorSpecies<Float> S = FloatVector.SPECIES_PREFERRED;

        @Override
        public float applyAsFloat(float a, float b) {
            return a * b;
        }

        @Override
        public void binaryBroadcastScalar(MemorySegment a, float b, MemorySegment r, long offset, long length) {
            // Slice to columns
            a = a.asSlice(offset << 2, length << 2);
            r = r.asSlice(offset << 2, length << 2);

            // Perform vector loop
            int iLength = (int) length;
            int i = 0;
            // Broadcast b into a vector
            FloatVector bV = FloatVector.broadcast(S, b);
            for (; i < S.loopBound(iLength); i += S.length()) {
                // Load vector
                FloatVector aV = FloatVector.fromMemorySegment(S, a, ((long) i) << 2, ByteOrder.nativeOrder());
                // Data parallel lanewise multiply
                FloatVector rV = aV.mul(bV);
                // Store vector
                rV.intoMemorySegment(r, ((long) i) << 2, ByteOrder.nativeOrder());
            }

            // Perform scalar
            for (; i < length; i++) {
                float aE = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float rE = aE * b;
                r.setAtIndex(ValueLayout.JAVA_FLOAT, i, rE);
            }
        }

        void x() {
            FloatBinaryOperator f = (float a, float b) -> a * b;
            BinaryOperator<FloatVector> fv = (FloatVector aV, FloatVector bV) -> aV.mul(bV);
        }
    }

    @Test
    public void testUsingKernel() {
        var m = FloatMatrix.newFloatMatrix(17, 3);
        m.setEach((i, j, v) -> j * m.rows() + i);
        m.print();

        var rowVector = FloatMatrix.newFloatMatrix(1, 3);
        rowVector.setEach((i, j, v) -> j);
        rowVector.print();

        m.elementwise(rowVector, new VectorMulKernel());
        m.print();
    }
}

/*
m[10000, 1000]
rowVector[1, 1000]

Benchmark                         Mode  Cnt        Score       Error  Units
KernelFloat.scalar                avgt   10  4666300.803 ± 26954.183  ns/op
KernelFloat.scalarKernel          avgt   10  1695387.821 ±  8885.760  ns/op
KernelFloat.vectorKernel          avgt   10   645317.219 ± 11414.014  ns/op
KernelFloat.vectorKernelParallel  avgt   10   268572.661 ±  2625.970  ns/op

*/
