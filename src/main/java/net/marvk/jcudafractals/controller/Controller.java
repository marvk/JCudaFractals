package net.marvk.jcudafractals.controller;

import net.marvk.jcudafractals.fractal.Buddhabrot2;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * Created by Marvin on 25.05.2016.
 */
public final class Controller {
    private static final int SIZE = ((int) Math.pow(2, 12));
    private static final long NUM_POINTS = 128L * 128L * 128L * 128L * 2L;

    public Controller() {
        final int[] iterations = {10000*5, 2000*3, 400*2};

        final long startTime = System.currentTimeMillis();
        final int[][] result = IntStream.range(0, 3)
                                        .mapToObj(i -> Buddhabrot2.buddhabrot(iterations[i], NUM_POINTS, SIZE))
                                        .toArray(int[][]::new);

        System.out.println("Seconds elapsed: " + (System.currentTimeMillis() - startTime) / 1000);

        final BufferedImage image = new BufferedImage(SIZE, SIZE, BufferedImage.TYPE_INT_RGB);

        final double[] maxValues =
                IntStream.range(0, 3)
                         .mapToDouble(i -> Arrays.stream(result[i]).max().orElse(Integer.MAX_VALUE) / 255.0)
                         .toArray();

        final AtomicInteger cnt = new AtomicInteger();

        IntStream.range(0, SIZE * SIZE)
                 .parallel()
                 .forEach(i -> {
                     if (cnt.incrementAndGet() % 100_000 == 0) {
                         System.out.println(cnt.get() / (double) (SIZE * SIZE));
                     }
                     final int x = i % SIZE;
                     final int y = i / SIZE;
                     final int r = (int) (result[0][i] / maxValues[0]);
                     final int g = (int) (result[1][i] / maxValues[1]);
                     final int b = (int) (result[2][i] / maxValues[2]);

                     image.setRGB(y, x, 0xFF000000 | (r << 16) | (g << 8) | b);
                 });

        try {
            ImageIO.write(image, "bmp", new File("nebula" + System.currentTimeMillis() + ".bmp"));
        } catch (final IOException e) {
            e.printStackTrace();
        }
    }
}
