package jmh;

import oracle.blis.matrix.FloatMatrix;
import org.openjdk.jmh.annotations.*;

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
public class ScalarFloatBench {

    @Param({"false", "true"}) String pollute;

    FloatMatrix A;
    FloatMatrix B;
    FloatMatrix C;

    @Setup
    public void setup() {
        A = FloatMatrix.newFloatMatrix(1000, 100);
        A.randm();

        B = FloatMatrix.newFloatMatrix(1, 100);
        B.randm();

        C = FloatMatrix.newFloatMatrix(1000, 100);

        // pollute
        if (Boolean.valueOf(pollute)) {
            A.elementwise(B, C, (a, b) -> a * b);
            A.elementwise(B, C, (a, b) -> a * b);
        }
    }

    @Benchmark
    public void scalar() {
        // A[1000, 100] * B[1, 100]
        A.elementwise(B, C, (a, b) -> a * b);
    }
}
