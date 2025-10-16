package jmh;

import jdk.incubator.vector.*;

public class Atan {

    /*
https://github.com/jeremybarnes/cephes/blob/master/single/atanf.c#L85
     */

    static final int SIGN_MASK_I = 0x80000000;
    static final int INV_SIGN_MASK_I = ~0x80000000;

    static final double HALF_PI = Math.PI / 2.0;
    static final double QUATER_PI = Math.PI / 4.0;
    static final float HALF_PI_F = (float) HALF_PI;
    static final float QUATER_PI_F = (float) QUATER_PI;

    // Math.tan(3.0 * Math.PI / 8.0)
    static final double TAN_3_PI_OVER_8 = 2.41421356237309504880;

    static final float ATANF_RANGE_HI_F = (float) TAN_3_PI_OVER_8;
    static final float ATANF_RANGE_LOW_F = 0.4142135623730950f;

    static final float ATANF_COF_P0_F = 8.05374449538e-2f;
    static final float ATANF_COF_P1_F = 1.38776856032e-1f;
    static final float ATANF_COF_P2_F = 1.99777106478e-1f;
    static final float ATANF_COF_P3_F = 3.33329491539e-1f;

    static final VectorSpecies<Float> SPECIES_F = FloatVector.SPECIES_PREFERRED;

    public static float atanf(float xx) {
        float x, y, z;
        int sign;

        x = xx;

        /* make argument positive and save the sign */
        if (xx < 0.0f) {
            sign = -1;
            x = -xx;
        } else {
            sign = 1;
            x = xx;
        }
        /* range reduction */
        if (x > ATANF_RANGE_HI_F)  /* tan 3pi/8 */ {
            y = HALF_PI_F;
            x = -(1.0f / x);
        } else if (x > ATANF_RANGE_LOW_F) /* tan pi/8 */ {
            y = QUATER_PI_F;
            x = (x - 1.0f) / (x + 1.0f);
        } else
            y = 0.0f;

        z = x * x;
        y +=
                (((8.05374449538e-2f * z
                        - 1.38776856032e-1f) * z
                        + 1.99777106478e-1f) * z
                        - 3.33329491539e-1f) * z * x
                        + x;

        if (sign < 0)
            y = -y;

        return y;
    }

    public static FloatVector atanf(FloatVector x) {
        IntVector sign_bit = x.reinterpretAsInts();
        // Take the absolute value
        x = sign_bit.and(INV_SIGN_MASK_I).reinterpretAsFloats();
        // Extract the sign bit (upper one) to be applied to the result
        sign_bit = sign_bit.and(SIGN_MASK_I);

        // x > 2.414213562373095
        VectorMask<Float> cmpHi = x.compare(VectorOperators.GT, ATANF_RANGE_HI_F);
        // x > 0.4142135623730950
        VectorMask<Float> cmpLow = x.compare(VectorOperators.GT, ATANF_RANGE_LOW_F);
        // x > 0.4142135623730950 && !( x > 2.414213562373095 )
        VectorMask<Float> cmpLowHigh = cmpLow.andNot(cmpHi);

        // x0 = -1.0 / x
        FloatVector x0 = FloatVector.broadcast(SPECIES_F, -1.0f).div(x);

        // x1 = (x - 1.0) / (x + 1.0)
        FloatVector x1 = x.sub(1.0f).div(x.add(1.0f));

        // x0 if x > 2.414213562373095
        x = x.blend(x0, cmpHi);
        // x1 if x > 0.4142135623730950 && !( x > 2.414213562373095 )
        x = x.blend(x1, cmpLowHigh);
        // x if x <= 0.4142135623730950

        // PI / 2 if x > 2.414213562373095
        FloatVector y0 = FloatVector.zero(SPECIES_F).blend(HALF_PI_F, cmpHi);
        // PI / 4 if x > 0.4142135623730950 && !( x > 2.414213562373095 )
        FloatVector y = y0.blend(QUATER_PI_F, cmpLowHigh);
        // 0.0 if x <= 0.4142135623730950

        FloatVector zz = x.mul(x);
        FloatVector acc = zz.fma(
                FloatVector.broadcast(SPECIES_F, ATANF_COF_P0_F),
                FloatVector.broadcast(SPECIES_F, -ATANF_COF_P1_F));
        acc = acc.fma(
                zz,
                FloatVector.broadcast(SPECIES_F, ATANF_COF_P2_F));
        acc = acc.fma(
                zz,
                FloatVector.broadcast(SPECIES_F, -ATANF_COF_P3_F));
        acc = acc.mul(zz);
        acc = acc.fma(x, x);
        y = y.add(acc);

        // Update the sign
        y = y.reinterpretAsInts().lanewise(VectorOperators.XOR, sign_bit).reinterpretAsFloats();

        return y;
    }
}
