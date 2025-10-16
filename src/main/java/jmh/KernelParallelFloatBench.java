package jmh;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import oracle.blis.matrix.FloatMatrix;
import oracle.blis.matrix.kernel.FloatBroadcastKernel;
import org.openjdk.jmh.annotations.*;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread) // Startup / Teardown / time done with single thread
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 10, time = 1)
@Fork(value = 1, jvmArgsPrepend =
        {"-Djava.library.path=/Users/sandoz/Projects/jdk/blis/blis-flame/lib/firestorm",
                "--enable-native-access=ALL-UNNAMED",
                "--add-modules=jdk.incubator.vector",
                "--enable-preview"})
public class KernelParallelFloatBench {

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
            long i = 0;
            FloatVector bV = FloatVector.broadcast(S, b);
            for (; i < S.loopBound(length); i += S.length()) {
                // Load vector
                FloatVector aV = FloatVector.fromMemorySegment(S, a, i << 2, ByteOrder.nativeOrder());
                FloatVector rV = aV.mul(bV);
                // Store vector
                rV.intoMemorySegment(r, i << 2, ByteOrder.nativeOrder());
            }

            // Perform scalar
            for (; i < length; i++) {
                float aE = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float rE = aE * b;
                r.setAtIndex(ValueLayout.JAVA_FLOAT, i, rE);
            }
        }
    }

    static class VectorMulKernelParallel extends VectorMulKernel {
        @Override
        public void binaryBroadcastVector(MemorySegment a, MemorySegment b, MemorySegment r, long aStride, long bLength) {
            IntStream.range(0, (int) bLength).parallel().forEach(j -> {
                float bE = b.getAtIndex(ValueLayout.JAVA_FLOAT, j);
                binaryBroadcastScalar(a, bE, r, j * aStride, aStride);
            });
        }
    }

    @Param({"false", "true"}) String pollute;

    FloatMatrix A;
    FloatMatrix B;
    FloatMatrix C;
    VectorMulKernel vectorKernel;
    VectorMulKernelParallel vectorKernelParallel;

    @Setup
    public void setup() {
        A = FloatMatrix.newFloatMatrix(10000, 1000);
        A.randm();

        B = FloatMatrix.newFloatMatrix(1, 1000);
        B.randm();

        C = FloatMatrix.newFloatMatrix(10000, 1000);

        vectorKernel = new VectorMulKernel();
        vectorKernelParallel = new VectorMulKernelParallel();
    }

    @Benchmark
    public void vectorKernel() {
        A.elementwise(B, C, vectorKernel);
    }

    @Benchmark
    public void vectorKernelParallel() {
        A.elementwise(B, C, vectorKernelParallel);
    }
}
