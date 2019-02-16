package net.marvk.jcudafractals.fractal;

/**
 * Created by Marvin on 25.05.2016.
 */
public class FractalHelper {
    private FractalHelper() {
        //no instance
    }

    public static boolean isInCardiodicOrP2Bulb(float cR, float cI) {
        float q = ((cR - (1f / 4f)) * (cR - (1f / 4f))) + (cI * cI);

        return q * (q + (cR - (1f / 4f))) < (1f / 4f) * (cI * cI)  //Check if point is within the cardiodic
                || (cR + 1f) * (cR + 1f) + (cI * cI) < (1f / 16f);   //Check if point is within the period-2 bulb


    }
}
