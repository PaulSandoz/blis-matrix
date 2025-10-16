package jmh;

import jdk.incubator.code.*;
import jdk.incubator.code.analysis.Inliner;
import jdk.incubator.code.analysis.SSA;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import oracle.blis.matrix.FloatMatrix;
import oracle.blis.matrix.f.FloatBinaryOperator;
import oracle.blis.matrix.kernel.FloatBroadcastKernel;
import org.openjdk.jmh.annotations.*;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.nio.ByteOrder;
import java.util.Collections;
import java.util.concurrent.TimeUnit;

import static java.nio.ByteOrder.nativeOrder;

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
public class ReflectableKernelFloatBench {

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

    @FunctionalInterface
    interface ReflectableFloatBinaryOperator extends FloatBinaryOperator, Quotable {
    }

    static class ReflectableVectorKernel implements FloatBroadcastKernel.BinaryBroadcastKernel {
        static final VectorSpecies<Float> S = FloatVector.SPECIES_PREFERRED;

        final MethodHandle applyAsFloat_MH;
        final MethodHandle binaryBroadcastScalar_MH;

        ReflectableVectorKernel(ReflectableFloatBinaryOperator o) throws Exception {
            Method tMethod = ReflectableVectorKernel.class.getDeclaredMethod("binaryBroadcastScalar_T",
                    MemorySegment.class, float.class, MemorySegment.class, long.class, long.class,
                    VectorSpecies.class);

            CoreOp.FuncOp tCodeModel = Op.ofMethod(tMethod).orElseThrow();
            System.out.println(tCodeModel.toText());

            Quoted quoted = Op.ofQuotable(o).orElseThrow();
            if (!quoted.capturedValues().isEmpty()) {
                throw new IllegalArgumentException();
            }
            JavaOp.LambdaOp scalarOpCodeModel = (JavaOp.LambdaOp) quoted.op();
            System.out.println(scalarOpCodeModel.toText());
            this.applyAsFloat_MH = BytecodeGenerator.generate(MethodHandles.lookup(), scalarOpCodeModel);

            CoreOp.FuncOp vectorOpCodeModel = scalarToVectorTransform(scalarOpCodeModel);
            System.out.println(vectorOpCodeModel.toText());

            CoreOp.FuncOp sCodeModel = tCodeModel.transform((block, op) -> {
                switch (op) {
                    case JavaOp.InvokeOp iOp when iOp.invokeDescriptor().name().equals("scalar_T") -> Inliner.inline(
                            block,
                            SSA.transform(scalarOpCodeModel),
                            block.context().getValues(op.operands()),
                            (_, v) -> block.context().mapValue(op.result(), v));
                    case JavaOp.InvokeOp iOp when iOp.invokeDescriptor().name().equals("vector_T") -> Inliner.inline(
                            block,
                            SSA.transform(vectorOpCodeModel),
                            block.context().getValues(op.operands()),
                            (_, v) -> block.context().mapValue(op.result(), v));
                    default -> block.op(op);
                }
                return block;
            });
            System.out.println(sCodeModel.toText());
            CoreOp.FuncOp lsCodeModel = sCodeModel.transform(OpTransformer.LOWERING_TRANSFORMER);
            this.binaryBroadcastScalar_MH = BytecodeGenerator.generate(MethodHandles.lookup(), lsCodeModel);
        }

        static CoreOp.FuncOp scalarToVectorTransform(JavaOp.LambdaOp scalar) {
            FunctionType fType = CoreType.functionType(
                    JavaType.type(FloatVector.class),
                    Collections.nCopies(scalar.invokableType().parameterTypes().size(),
                            JavaType.type(FloatVector.class)));

            return CoreOp.func("vector", fType).body(fblock -> {
                fblock.body(scalar.body(), fblock.parameters(), (block, op) -> {
                    if (op instanceof JavaOp.MulOp) {
                        var result = block.op(JavaOp.invoke(
                                MethodRef.method(FloatVector.class, "mul", FloatVector.class, FloatVector.class),
                                block.context().getValues(op.operands())));
                        block.context().mapValue(op.result(), result);
                    } else {
                        block.op(op);
                    }
                    return block;
                });
            });
        }

        @Override
        public float applyAsFloat(float a, float b) {
            try {
                return (float) applyAsFloat_MH.invokeExact(a, b);
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        public void binaryBroadcastScalar(MemorySegment a, float b, MemorySegment r,
                                          long offset, long length) {
            try {
                binaryBroadcastScalar_MH.invokeExact(a, b, r, offset, length, S);
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
        }


        @CodeReflection
        private static void binaryBroadcastScalar_T(MemorySegment a, float b,
                                                    MemorySegment r,
                                                    long offset, long length,
                                                    VectorSpecies<Float> S) {
            a = a.asSlice(offset << 2, length << 2);
            r = r.asSlice(offset << 2, length << 2);

            long i = 0;
            var bV = FloatVector.broadcast(S, b);
            for (; i < S.loopBound(i); i += S.length()) {
                var aV = FloatVector.fromMemorySegment(S, a, i << 2, nativeOrder());
                var rV = vector_T(aV, bV);
                rV.intoMemorySegment(r, i << 2, nativeOrder());
            }

            for (; i < length; i++) {
                float aE = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                float rE = scalar_T(aE, b);
                r.setAtIndex(ValueLayout.JAVA_FLOAT, i, rE);
            }
        }


        private static float scalar_T(float a, float b) {
            throw new UnsupportedOperationException();
        }

        private static FloatVector vector_T(FloatVector a, FloatVector b) {
            throw new UnsupportedOperationException();
        }
    }

    FloatMatrix A;
    FloatMatrix B;
    FloatMatrix C;
    VectorMulKernel vectorMulKernel;
    ReflectableVectorKernel reflectableMulKernel;

    @Setup
    public void setup() throws Exception {
        A = FloatMatrix.newFloatMatrix(1000, 100);
        A.randm();

        B = FloatMatrix.newFloatMatrix(1, 100);
        B.randm();

        C = FloatMatrix.newFloatMatrix(1000, 100);

        vectorMulKernel = new VectorMulKernel();
        reflectableMulKernel = new ReflectableVectorKernel((a, b) -> a * b);
    }

    @Benchmark
    public void scalar() {
        A.elementwise(B, C, (a, b) -> a * b);
    }

    @Benchmark
    public void vectorKernel() {
        A.elementwise(B, C, vectorMulKernel);
    }

    @Benchmark
    public void reflectableVectorKernel() {
        A.elementwise(B, C, reflectableMulKernel);
    }
}
