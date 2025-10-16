package oracle.blis.matrix;

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
import oracle.blis.matrix.f.FloatBinaryOperator;
import oracle.blis.matrix.kernel.FloatBroadcastKernel;
import org.junit.jupiter.api.Test;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.nio.ByteOrder;
import java.util.Collections;

import static java.nio.ByteOrder.*;

public class TestKernelCodeReflection {

    @FunctionalInterface
    interface ReflectableFloatBinaryOperator extends FloatBinaryOperator, Quotable {
    }


    static class ReflectableScalarKernel implements FloatBroadcastKernel.BinaryBroadcastKernel {
        final MethodHandle applyAsFloat_MH;
        final MethodHandle binaryBroadcastScalar_MH;

        ReflectableScalarKernel(ReflectableFloatBinaryOperator o) throws Exception {
            Method tMethod = ReflectableScalarKernel.class.getDeclaredMethod("binaryBroadcastScalar_T",
                    MemorySegment.class, float.class, MemorySegment.class, long.class, long.class);

            CoreOp.FuncOp tCodeModel = Op.ofMethod(tMethod).orElseThrow();
            System.out.println(tCodeModel.toText());

            Quoted quoted = Op.ofQuotable(o).orElseThrow();
            if (!quoted.capturedValues().isEmpty()) {
                throw new IllegalArgumentException();
            }
            JavaOp.LambdaOp lCodeModel = (JavaOp.LambdaOp) quoted.op();
            System.out.println(lCodeModel.toText());
            this.applyAsFloat_MH = BytecodeGenerator.generate(MethodHandles.lookup(), lCodeModel);

            CoreOp.FuncOp sCodeModel = tCodeModel.transform((block, op) -> {
                if (op instanceof JavaOp.InvokeOp iOp && iOp.invokeDescriptor().name().equals("scalar_T")) {
                    Inliner.inline(
                            block,
                            lCodeModel,
                            block.context().getValues(op.operands()),
                            (b, v) -> {
                                block.context().mapValue(op.result(), v);
                            });
                    return block;
                } else {
                    block.op(op);
                    return block;
                }
            });
            System.out.println(sCodeModel.toText());

            CoreOp.FuncOp lsCodeModel = sCodeModel.transform(OpTransformer.LOWERING_TRANSFORMER);
            this.binaryBroadcastScalar_MH = BytecodeGenerator.generate(MethodHandles.lookup(), lsCodeModel);
        }

        @Override
        public float applyAsFloat(float a, float b) {
            try {
                return (float) applyAsFloat_MH.invokeExact(a, b);
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
        }

        public void binaryBroadcastScalar(MemorySegment a, float b, MemorySegment r,
                                          long offset, long length) {
            try {
                binaryBroadcastScalar_MH.invokeExact(a, b, r, offset, length);
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
        }


        @CodeReflection
        private static void binaryBroadcastScalar_T(MemorySegment a, float b, MemorySegment r,
                                                    long offset, long length) {
            for (long i = 0; i < length; i++) {
                float aE = a.getAtIndex(ValueLayout.JAVA_FLOAT, offset + i);
                float rE = scalar_T(aE, b);
                r.setAtIndex(ValueLayout.JAVA_FLOAT, offset + i, rE);
            }
        }

        private static float scalar_T(float a, float b) {
            throw new UnsupportedOperationException();
        }
    }

    @Test
    public void testReflectableScalarKernel() throws Exception {
        var kernel = new ReflectableScalarKernel((a, b) -> a * b);

        var m = FloatMatrix.newFloatMatrix(5, 3);
        m.setEach((i, j, v) -> j * m.rows() + i);
        m.print();

        var rowVector = FloatMatrix.newFloatMatrix(1, 3);
        rowVector.setEach((i, j, v) -> j);
        rowVector.print();

        m.elementwise(rowVector, kernel);
        m.print();
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

    @Test
    public void testReflectableVectorKernel() throws Exception {
        var kernel = new ReflectableVectorKernel((a, b) -> a * b);

        var m = FloatMatrix.newFloatMatrix(5, 3);
        m.setEach((i, j, v) -> j * m.rows() + i);
        m.print();

        var rowVector = FloatMatrix.newFloatMatrix(1, 3);
        rowVector.setEach((i, j, v) -> j);
        rowVector.print();

//        m.elementwise(rowVector, kernel);
        m.elementwise(rowVector, (a, b) -> a * b);
        m.print();
    }
}