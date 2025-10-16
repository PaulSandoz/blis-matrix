package jmh;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import oracle.blis.matrix.kernel.FloatBroadcastKernel;
import oracle.blis.matrix.FloatMatrix;
import org.openjdk.jmh.annotations.*;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread) // Startup / Teardown / time done with single thread
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 10, time = 1)
@Fork(value = 1, jvmArgsPrepend =
        {"-Djava.library.path=/Users/sandoz/Projects/jdk/blis/blis-flame/lib/firestorm",
                "--enable-native-access=ALL-UNNAMED",
                "--add-modules=jdk.incubator.vector",
                "--enable-preview"})
public class KernelFloatBench {

    static class ScalarMulKernel implements FloatBroadcastKernel.BinaryBroadcastKernel {
        @Override
        public float applyAsFloat(float a, float b) {
            return a * b;
        }

        @Override
        public void binaryBroadcastScalar(MemorySegment A, float b, MemorySegment C, long offset, long length) {
            // Slice to columns
            A = A.asSlice(offset << 2, length << 2);
            C = C.asSlice(offset << 2, length << 2);

            // Perform scalar
            for (long i = 0; i < length; i++) {
                float a = A.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float c = a * b;
                C.setAtIndex(ValueLayout.JAVA_FLOAT, i, c);
            }
        }
    }

    static class VectorMulKernel implements FloatBroadcastKernel.BinaryBroadcastKernel {
        static final VectorSpecies<Float> S = FloatVector.SPECIES_PREFERRED;

        @Override
        public float applyAsFloat(float a, float b) {
            return a * b;
        }

        @Override
        public void binaryBroadcastScalar(MemorySegment A, float b, MemorySegment C, long offset, long length) {
            // Slice to columns
            A = A.asSlice(offset << 2, length << 2);
            C = C.asSlice(offset << 2, length << 2);

            // Perform vector loop
            long i = 0;
            FloatVector bV = FloatVector.broadcast(S, b);
            for (; i < S.loopBound(length); i += S.length()) {
                // Load vector
                FloatVector aV = FloatVector.fromMemorySegment(S, A, i << 2, ByteOrder.nativeOrder());
                FloatVector cV = aV.mul(bV);
                // Store vector
                cV.intoMemorySegment(C, i << 2, ByteOrder.nativeOrder());
            }

            // Perform scalar
            for (; i < length; i++) {
                float a = A.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float c = a * b;
                C.setAtIndex(ValueLayout.JAVA_FLOAT, i, c);
            }
        }
    }

    @Param({"false", "true"}) String pollute;

    FloatMatrix A;
    FloatMatrix B;
    FloatMatrix C;
    ScalarMulKernel scalarMulKernel;
    VectorMulKernel vectorMulKernel;

    @Setup
    public void setup() {
        A = FloatMatrix.newFloatMatrix(1000, 100);
        A.randm();

        B = FloatMatrix.newFloatMatrix(1, 100);
        B.randm();

        C = FloatMatrix.newFloatMatrix(1000, 100);

        scalarMulKernel = new ScalarMulKernel();
        vectorMulKernel = new VectorMulKernel();

        if (Boolean.valueOf(pollute)) {
            A.elementwise(B, C, (a, b) -> a * b);
            A.elementwise(B, C, (a, b) -> a * b);
        }
    }

    @Benchmark
    public void scalar() {
        A.elementwise(B, C, (a, b) -> a * b);
    }

    @Benchmark
    public void scalarKernel() {
        A.elementwise(B, C, scalarMulKernel);
    }

    @Benchmark
    public void vectorKernel() {
        A.elementwise(B, C, vectorMulKernel);
    }
}
