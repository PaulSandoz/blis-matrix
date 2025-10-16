package jmh;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import oracle.blis.matrix.FloatMatrix;
import oracle.blis.matrix.kernel.FloatBroadcastKernel;
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
public class AtanFloatBench {

    static class VectorCephesAtanKernel implements FloatBroadcastKernel.BinaryBroadcastKernel {
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

                FloatVector cV = Atan.atanf(aV).mul(bV);

                // Store vector
                cV.intoMemorySegment(C, i << 2, ByteOrder.nativeOrder());
            }

            // Perform scalar
            for (; i < length; i++) {
                float a = A.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float c = Atan.atanf(a) * b;
                C.setAtIndex(ValueLayout.JAVA_FLOAT, i, c);
            }
        }
    }

    static class VectorMathAtanKernel implements FloatBroadcastKernel.BinaryBroadcastKernel {
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

                FloatVector cV = aV.lanewise(VectorOperators.ATAN).mul(bV);

                // Store vector
                cV.intoMemorySegment(C, i << 2, ByteOrder.nativeOrder());
            }

            // Perform scalar
            for (; i < length; i++) {
                float a = A.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float c = ((float) Math.atan(a)) * b;
                C.setAtIndex(ValueLayout.JAVA_FLOAT, i, c);
            }
        }
    }

    FloatMatrix A;
    FloatMatrix B;
    FloatMatrix C;
    VectorCephesAtanKernel vectorCephesKernel;
    VectorMathAtanKernel vectorMathKernel;

    @Setup
    public void setup() {
        A = FloatMatrix.newFloatMatrix(1000, 100);
        A.randm();

        B = FloatMatrix.newFloatMatrix(1, 100);
        B.randm();

        C = FloatMatrix.newFloatMatrix(1000, 100);

        vectorCephesKernel = new VectorCephesAtanKernel();
        vectorMathKernel = new VectorMathAtanKernel();
    }

    @Benchmark
    public void scalarCephes() {
        // Not polluted
        A.elementwise(B, C,
                (a, b) -> Atan.atanf(a) * b);
    }

    @Benchmark
    public void scalarMath() {
        // Not polluted
        A.elementwise(B, C,
                (a, b) -> ((float) Math.atan(a)) * b);
    }

    @Benchmark
    public void vectorCephesKernel() {
        // FloatVector cV = Atan.atanf(aV).mul(bV)
        A.elementwise(B, C, vectorCephesKernel);
    }

    @Benchmark
    public void vectorMathKernel() {
        // FloatVector cV = aV.lanewise(VectorOperators.ATAN)
        //                    .mul(bV)
        A.elementwise(B, C, vectorMathKernel);
    }
}
