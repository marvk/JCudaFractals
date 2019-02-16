package net.marvk.jcudafractals.fractal;

/**
 * Created by Marvin on 25.05.2016.
 */
public final class FractalHelper {
    private FractalHelper() {
        throw new AssertionError("No instances of utility class " + FractalHelper.class);
    }

    public static boolean isInCardiodicOrP2Bulb(final float cR, final float cI) {
        final float q = ((cR - (1f / 4f)) * (cR - (1f / 4f))) + (cI * cI);

        return q * (q + (cR - (1f / 4f))) < (1f / 4f) * (cI * cI)  //Check if point is within the cardiodic
                || (cR + 1f) * (cR + 1f) + (cI * cI) < (1f / 16f);   //Check if point is within the period-2 bulb

    }
}
