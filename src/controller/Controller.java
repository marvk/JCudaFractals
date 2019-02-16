package controller;

import fractal.Buddhabrot2;

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
public class Controller {
    public static final int SIZE = 16384;
    public static long NUM_POINTS = 128L * 128L * 128L * 128L;

    public Controller() {
        int[] iterations = {10000, 2000, 400};

        int[][] result = new int[3][];

        long startTime = System.currentTimeMillis();

        for (int i = 0; i < 3; i++) {
            result[i] = Buddhabrot2.buddhabrot(iterations[i], NUM_POINTS, SIZE);
        }

        System.out.println("Seconds elapsed: " + (System.currentTimeMillis() - startTime) / 1000);

        BufferedImage image = new BufferedImage(SIZE, SIZE, BufferedImage.TYPE_INT_RGB);

//        for (int i = 2046 * 64; i < 2047 * 64; i++) {
//            System.out.println(Arrays.toString(Arrays.copyOfRange(outX, ITERATIONS * i, ITERATIONS * (i + 1))));
//            System.out.println(Arrays.toString(Arrays.copyOfRange(outY, ITERATIONS * i, ITERATIONS * (i + 1))));
//            System.out.println();
//        }

        double[] maxValues = new double[3];

        for (int i = 0; i < 3; i++) {
            maxValues[i] = Arrays.stream(result[i]).max().orElse(Integer.MAX_VALUE) / 255.0;
        }

        AtomicInteger cnt = new AtomicInteger();

        IntStream.range(0, SIZE * SIZE)
                 .parallel()
                 .forEach(i -> {
                     if (cnt.incrementAndGet() % 100_000 == 0) {
                         System.out.println(((double) cnt.get()) / (double) (SIZE * SIZE));
                     }
                     int x = i % SIZE;
                     int y = i / SIZE;
                     int r = (int) (result[0][i] / maxValues[0]);
                     int g = (int) (result[1][i] / maxValues[1]);
                     int b = (int) (result[2][i] / maxValues[2]);

                     image.setRGB(y, x, 0xFF000000 | (r << 16) | (g << 8) | b);
                 });

        try {
            ImageIO.write(image, "bmp", new File("nebula" + System.currentTimeMillis() + ".bmp"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
