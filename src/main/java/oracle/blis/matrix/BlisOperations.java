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

import java.lang.annotation.Target;

import static java.lang.annotation.ElementType.METHOD;

public final class BlisOperations {

    static final BlisOperations INSTANCE = new BlisOperations();

    public static BlisOperations singleton() {
        return INSTANCE;
    }

    private BlisOperations() {
    }

    // Operations

    @Target(METHOD)
    @interface Expression {
        String value();
    }

    // Argument checks required
    // - Check dimensions
    // - Check if argument to update is a constant

    // Level-1v operations

    @Expression("B := B + trans?(A)")
    public void addv(Matrix<?> x, Matrix<?> y) {
        blis_h.bli_addv(x.obj, y.obj);
    }

    public void amaxv(Matrix<?> x, Matrix<?> index) {
        blis_h.bli_amaxv(x.obj, index.obj);
    }

    @Expression("y := y + conj?(alpha) * conj?(x)")
    public void axpyv(Matrix<?> alpha, Matrix<?> x, Matrix<?> y) {
        blis_h.bli_axpyv(alpha.obj, x.obj, y.obj);
    }

    @Expression("y := conj?(beta) * y + conj?(alpha) * conj?(x)")
    public void axpbyv(Matrix<?> alpha, Matrix<?> x, Matrix<?> beta, Matrix<?> y) {
        blis_h.bli_axpbyv(alpha.obj, x.obj, beta.obj, y.obj);
    }

    @Expression("y := conj?(x)")
    public void copyv(Matrix<?> x, Matrix<?> y) {
        blis_h.bli_copyv(x.obj, y.obj);
    }

    @Expression("rho := conj?(x)^T * conj?(y)")
    public void dotv(Matrix<?> x, Matrix<?> y, Matrix<?> rho) {
        blis_h.bli_dotv(x.obj, y.obj, rho.obj);
    }

    @Expression("rho := conj?(x)^T * conj?(y)")
    public void dotxv(Matrix<?> alpha, Matrix<?> x, Matrix<?> y, Matrix<?> beta, Matrix<?> rho) {
        blis_h.bli_dotxv(alpha.obj, x.obj, y.obj, beta.obj, rho.obj);
    }

    public void bli_invertv(Matrix<?> x) {
        blis_h.bli_invertv(x.obj);
    }

    @Expression("x := conj?(alpha) * x")
    public void scalv(Matrix<?> alpha, Matrix<?> x) {
        blis_h.bli_scalv(alpha.obj, x.obj);
    }

    @Expression("y := conj?(alpha) * conj?(x)")
    public void scal2v(Matrix<?> alpha, Matrix<?> x, Matrix<?> y) {
        blis_h.bli_scal2v(alpha.obj, x.obj, y.obj);
    }

    @Expression("x := conj?(alpha)")
    public void setv(Matrix<?> alpha, Matrix<?> x) {
        blis_h.bli_setv(alpha.obj, x.obj);
    }

    @Expression("real(x) := real(alpha)")
    public void setrv(Matrix<?> alpha, Matrix<?> x) {
        blis_h.bli_setrv(alpha.obj, x.obj);
    }

    @Expression("imag(x) := real(alpha)")
    public void setiv(Matrix<?> alpha, Matrix<?> x) {
        blis_h.bli_setiv(alpha.obj, x.obj);
    }

    @Expression("y := y - conj?(x)")
    public void subv(Matrix<?> x, Matrix<?> y) {
        blis_h.bli_subv(x.obj, y.obj);
    }

    public void swapv(Matrix<?> x, Matrix<?> y) {
        blis_h.bli_swapv(x.obj, y.obj);
    }

    @Expression("y := conj?(beta) * y + conj?(x)")
    public void xpbyv(Matrix<?> x, Matrix<?> beta, Matrix<?> y) {
        blis_h.bli_xpbyv(x.obj, beta.obj, y.obj);
    }


    // Level-1d operations

    @Expression("B := B + trans?(A)")
    public void addd(Matrix<?> a, Matrix<?> b) {
        blis_h.bli_addd(a.obj, b.obj);
    }

    @Expression("B := B + conj?(alpha) * trans?(A)")
    public void axpyd(Matrix<?> alpha, Matrix<?> a, Matrix<?> b) {
        blis_h.bli_axpyd(alpha.obj, a.obj, b.obj);
    }

    @Expression("B := trans?(A)")
    public void copyd(Matrix<?> a, Matrix<?> b) {
        blis_h.bli_copym(a.obj, b.obj);
    }

    public void invertd(Matrix<?> a) {
        blis_h.bli_invertd(a.obj);
    }

    @Expression("A := conj?(alpha) * A")
    public void scald(Matrix<?> alpha, Matrix<?> a) {
        blis_h.bli_scald(alpha.obj, a.obj);
    }

    @Expression("B := conj?(alpha) * trans?(A)")
    public void scal2d(Matrix<?> alpha, Matrix<?> a, Matrix<?> b) {
        blis_h.bli_scal2d(alpha.obj, a.obj, b.obj);
    }

    @Expression("A := conj?(alpha)")
    public void setd(Matrix<?> alpha, Matrix<?> a) {
        // check scalar otherwise exception
        blis_h.bli_setd(alpha.obj, a.obj);
    }

    @Expression("imag(A) := conj?(alpha)")
    public void setid(Matrix<?> alpha, Matrix<?> a) {
        // check scalar otherwise exception
        blis_h.bli_setid(alpha.obj, a.obj);
    }

    public void shiftd(Matrix<?> alpha, Matrix<?> a) {
        // check scalar otherwise exception
        blis_h.bli_shiftd(alpha.obj, a.obj);
    }

    @Expression("B := B - trans?(A)")
    public void subd(Matrix<?> a, Matrix<?> b) {
        blis_h.bli_subd(a.obj, b.obj);
    }

    @Expression("y := conj?(beta) * y + conj?(x)")
    public void xpbyd(Matrix<?> a, Matrix<?> beta, Matrix<?> b) {
        blis_h.bli_xpbyd(a.obj, beta.obj, b.obj);
    }


    // Level-1m operations

    @Expression("B := B + trans?(A)")
    public void addm(Matrix<?> a, Matrix<?> b) {
        blis_h.bli_addm(a.obj, b.obj);
    }

    @Expression("B := B + conj?(alpha) * trans?(A)")
    public void axpym(Matrix<?> alpha, Matrix<?> a, Matrix<?> b) {
        blis_h.bli_axpym(alpha.obj, a.obj, b.obj);
    }

    @Expression("B := trans?(A)")
    public void copym(Matrix<?> a, Matrix<?> b) {
        blis_h.bli_copym(a.obj, b.obj);
    }

    @Expression("A := conj?(alpha) * A")
    public void scalm(Matrix<?> alpha, Matrix<?> a) {
        blis_h.bli_scalm(alpha.obj, a.obj);
    }

    @Expression("B := conj?(alpha) * trans?(A)")
    public void scal2m(Matrix<?> alpha, Matrix<?> a, Matrix<?> b) {
        blis_h.bli_scal2m(alpha.obj, a.obj, b.obj);
    }

    @Expression("A := conj?(alpha)")
    public void setm(Matrix<?> alpha, Matrix<?> a) {
        // check scalar otherwise exception
        blis_h.bli_setm(alpha.obj, a.obj);
    }

    @Expression("real(A) := conj?(alpha)")
    public void setrm(Matrix<?> alpha, Matrix<?> a) {
        // check scalar otherwise exception
        blis_h.bli_setrm(alpha.obj, a.obj);
    }

    @Expression("imag(A) := conj?(alpha)")
    public void setim(Matrix<?> alpha, Matrix<?> a) {
        // check scalar otherwise exception
        blis_h.bli_setim(alpha.obj, a.obj);
    }

    @Expression("B := B - trans?(A)")
    public void subm(Matrix<?> a, Matrix<?> b) {
        blis_h.bli_subm(a.obj, b.obj);
    }


    // Level-1f operations

    @Expression("y := y + conj?(alphax) * conj?(x) + conj?(alphay) * conj?(y)")
    public void axpy2v(Matrix<?> alphax, Matrix<?> alphay, Matrix<?> x, Matrix<?> y, Matrix<?> z) {
        blis_h.bli_axpy2v(alphax.obj, alphay.obj, x.obj, y.obj, z.obj);
    }

    @Expression("rho := conj?(x)^T * conj?(y), y := y + conj?(alpha) * conj?(x)")
    public void dotaxpyv(Matrix<?> alpha, Matrix<?> xt, Matrix<?> x, Matrix<?> y, Matrix<?> rho, Matrix<?> z) {
        blis_h.bli_dotaxpyv(alpha.obj, xt.obj, x.obj, y.obj, rho.obj, z.obj);
    }

    @Expression("y := y + alpha * conja(A) * conjx(x)")
    public void axpyf(Matrix<?> alpha, Matrix<?> a, Matrix<?> x, Matrix<?> y) {
        blis_h.bli_axpyf(alpha.obj, a.obj, x.obj, y.obj);
    }

    @Expression("y := conj?(beta) * y + conj?(alpha) * conj?(A)^T * conj?(x)")
    public void dotxf(Matrix<?> alpha, Matrix<?> a, Matrix<?> x, Matrix<?> beta, Matrix<?> y) {
        blis_h.bli_dotxf(alpha.obj, a.obj, x.obj, beta.obj, y.obj);
    }

    @Expression("y := conj?(beta) * y + conj?(alpha) * conj?(A)^T * conj?(w), z := z + conj?(alpha) * conj?(A) * conj?(x)")
    public void dotxaxpyf(Matrix<?> alpha, Matrix<?> at, Matrix<?> a, Matrix<?> w, Matrix<?> x, Matrix<?> beta, Matrix<?> y, Matrix<?> z) {
        blis_h.bli_dotxaxpyf(alpha.obj, at.obj, a.obj, w.obj, x.obj, beta.obj, y.obj, z.obj);
    }


    // Level-2 operations

    @Expression("y := conj?(beta) * y + conj?(alpha) * trans?(A) * conj?(x)")
    public void gemv(Matrix<?> alpha, Matrix<?> a, Matrix<?> x, Matrix<?> beta, Matrix<?> y) {
        blis_h.bli_gemv(alpha.obj, a.obj, x.obj, beta.obj, y.obj);
    }

    @Expression("A := A + conj?(alpha) * conj?(x) * conj?(y)^T")
    public void ger(Matrix<?> alpha, Matrix<?> x, Matrix<?> y, Matrix<?> a) {
        blis_h.bli_ger(alpha.obj, x.obj, y.obj, a.obj);
    }

    @Expression("y := conj?(beta) * y + conj?(alpha) * conj?(A) * conj?(x)")
    public void hemv(Matrix<?> alpha, Matrix<?> a, Matrix<?> x, Matrix<?> beta, Matrix<?> y) {
        blis_h.bli_hemv(alpha.obj, a.obj, x.obj, beta.obj, y.obj);
    }

    @Expression("A := A + conj?(alpha) * conj?(x) * conj?(x)^H")
    public void her(Matrix<?> alpha, Matrix<?> x, Matrix<?> a) {
        blis_h.bli_her(alpha.obj, x.obj, a.obj);
    }

    @Expression("A := A + alpha * conj?(x) * conj?(y)^H + conj(alpha) * conj?(y) * conj?(x)^H")
    public void her2(Matrix<?> alpha, Matrix<?> x, Matrix<?> y, Matrix<?> a) {
        blis_h.bli_her2(alpha.obj, x.obj, y.obj, a.obj);
    }

    @Expression("y := conj?(beta) * y + conj?(alpha) * conj?(A) * conj?(x)")
    public void symv(Matrix<?> alpha, Matrix<?> a, Matrix<?> x, Matrix<?> beta, Matrix<?> y) {
        blis_h.bli_symv(alpha.obj, a.obj, x.obj, beta.obj, y.obj);
    }

    @Expression("A := A + conj?(alpha) * conj?(x) * conj?(x)^T")
    public void syr(Matrix<?> alpha, Matrix<?> x, Matrix<?> a) {
        blis_h.bli_syr(alpha.obj, x.obj, a.obj);
    }

    @Expression("A := A + alpha * conj?(x) * conj?(y)^T + conj(alpha) * conj?(y) * conj?(x)^T")
    public void syr2(Matrix<?> alpha, Matrix<?> x, Matrix<?> y, Matrix<?> a) {
        blis_h.bli_syr2(alpha.obj, x.obj, y.obj, a.obj);
    }

    @Expression("x := conj?(alpha) * transa(A) * x")
    public void trmv(Matrix<?> alpha, Matrix<?> x, Matrix<?> a) {
        blis_h.bli_trmv(alpha.obj, x.obj, a.obj);
    }

    @Expression("transa(A) * x = alpha * y")
    public void trsv(Matrix<?> alpha, Matrix<?> a, Matrix<?> y) {
        blis_h.bli_trsv(alpha.obj, a.obj, y.obj);
    }


    // Level-3 operations

    @Expression("C := beta * C + alpha * trans?(A) * trans?(B)")
    public void gemm(Matrix<?> alpha, Matrix<?> a, Matrix<?> b, Matrix<?> beta, Matrix<?> c) {
        blis_h.bli_gemm(alpha.obj, a.obj, b.obj, beta.obj, c.obj);
    }

    @Expression("C := beta * C + alpha * conj?(A) * trans?(B), C := beta * C + alpha * trans?(B) * conj?(A)")
    public void hemm(Matrix.Side sidea, Matrix<?> alpha, Matrix<?> a, Matrix<?> b, Matrix<?> beta, Matrix<?> c) {
        blis_h.bli_hemm(sidea.v, alpha.obj, a.obj, b.obj, beta.obj, c.obj);
    }

    @Expression("C := beta * C + alpha * trans?(A) * trans?(A)^H")
    public void herk(Matrix<?> alpha, Matrix<?> a, Matrix<?> beta, Matrix<?> c) {
        blis_h.bli_herk(alpha.obj, a.obj, beta.obj, c.obj);
    }

    @Expression("C := beta * C + alpha * trans?(A) * trans?(B)^H + conj(alpha) * trans?(B) * trans?(A)^H")
    public void her2k(Matrix<?> alpha, Matrix<?> a, Matrix<?> b, Matrix<?> beta, Matrix<?> c) {
        blis_h.bli_her2k(alpha.obj, a.obj, b.obj, beta.obj, c.obj);
    }

    @Expression("C := beta * C + alpha * conj?(A) * trans?(B), C := beta * C + alpha * trans?(B) * conj?(A)")
    public void symm(Matrix.Side sidea, Matrix<?> alpha, Matrix<?> a, Matrix<?> b, Matrix<?> beta, Matrix<?> c) {
        blis_h.bli_symm(sidea.v, alpha.obj, a.obj, b.obj, beta.obj, c.obj);
    }

    @Expression("C := beta * C + alpha * trans?(A) * trans?(A)^T")
    public void syrk(Matrix<?> alpha, Matrix<?> a, Matrix<?> beta, Matrix<?> c) {
        blis_h.bli_syrk(alpha.obj, a.obj, beta.obj, c.obj);
    }

    @Expression("C := beta * C + alpha * trans?(A) * trans?(B)^T + alpha * trans?(B) * trans?(A)^T")
    public void syr2k(Matrix<?> alpha, Matrix<?> a, Matrix<?> b, Matrix<?> beta, Matrix<?> c) {
        blis_h.bli_syr2k(alpha.obj, a.obj, b.obj, beta.obj, c.obj);
    }

    @Expression("B := alpha * transa(A) * B, B := alpha * B * transa(A)")
    public void trmm(Matrix.Side sidea, Matrix<?> alpha, Matrix<?> a, Matrix<?> b) {
        blis_h.bli_trmm(sidea.v, alpha.obj, a.obj, b.obj);
    }

    @Expression("C := beta * C + alpha * trans?(A) * trans?(B), C := beta * C + alpha * trans?(B) * trans?(A)")
    public void trmm3(Matrix.Side sidea, Matrix<?> alpha, Matrix<?> a, Matrix<?> b, Matrix<?> beta, Matrix<?> c) {
        blis_h.bli_trmm3(sidea.v, alpha.obj, a.obj, b.obj, beta.obj, c.obj);
    }

    @Expression("transa(A) * X = alpha * B, X * transa(A) = alpha * B")
    public void trsm(Matrix.Side sidea, Matrix<?> alpha, Matrix<?> a, Matrix<?> b) {
        blis_h.bli_trsm(sidea.v, alpha.obj, a.obj, b.obj);
    }


    // Utility operations
    // Possibly some directly on Matrix

    public void asumv(Matrix<?> x, Matrix<?> asum) {
        blis_h.bli_asumv(x.obj, asum.obj);
    }

    public void norm1m(Matrix<?> a, Matrix<?> norm) {
        blis_h.bli_norm1m(a.obj, norm.obj);
    }

    public void normfm(Matrix<?> a, Matrix<?> norm) {
        blis_h.bli_normfm(a.obj, norm.obj);
    }

    public void normim(Matrix<?> a, Matrix<?> norm) {
        blis_h.bli_normim(a.obj, norm.obj);
    }

    public void norm1v(Matrix<?> a, Matrix<?> norm) {
        blis_h.bli_norm1v(a.obj, norm.obj);
    }

    public void normfv(Matrix<?> a, Matrix<?> norm) {
        blis_h.bli_normfm(a.obj, norm.obj);
    }

    public void normiv(Matrix<?> a, Matrix<?> norm) {
        blis_h.bli_normim(a.obj, norm.obj);
    }

    public void mkherm(Matrix<?> a) {
        blis_h.bli_mkherm(a.obj);
    }

    public void mksymm(Matrix<?> a) {
        blis_h.bli_mksymm(a.obj);
    }

    public void mktrim(Matrix<?> a) {
        blis_h.bli_mktrim(a.obj);
    }

    public void randv(Matrix<?> a) {
        blis_h.bli_randv(a.obj);
    }

    public void randm(Matrix<?> a) {
        blis_h.bli_randm(a.obj);
    }

    public void sumsqv(Matrix<?> x, Matrix<?> scale, Matrix<?> sumsq) {
        blis_h.bli_sumsqv(x.obj, scale.obj, sumsq.obj);
    }
}
