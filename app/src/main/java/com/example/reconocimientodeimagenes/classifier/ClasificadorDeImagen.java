package com.example.reconocimientodeimagenes.classifier;

import android.app.Activity;
import android.graphics.Bitmap;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class ClasificadorDeImagen {
    /**
     * Los modelos de MobileNet cuantificados requieren una descuantificación adicional a la probabilidad de salida.
     */
    private static final float PROBABILITY_MEAN = 0.0f;
    private static final float PROBABILITY_STD = 255.0f;
    /**
     * El modelo cuantificado no requiere normalización, por lo tanto,
     * establezca la media en 0.0f y std en 1.0f para evitar la normalización.
     */
    private static final float IMAGE_STD = 1.0f;
    private static final float IMAGE_MEAN = 0.0f;
    private static final int MAX_SIZE = 5;
    /**
     * Tamaño de imagen a lo largo del eje x.
     */
    private final int imageResizeX;
    /**
     * Tamaño de imagen a lo largo del eje y.
     */
    private final int imageResizeY;
    /**
     * Etiquetas correspondientes a la salida del modelo de visión.
     */
    private final List<String> labels;
    /**
     *Una instancia de la clase de controlador para ejecutar la inferencia de modelos con Tensorflow Lite.
     */
    private final Interpreter tensorClassifier;
    /**
     * Imagen de entrada TensorBuffer.
     */
    private TensorImage inputImageBuffer;
    /**
     * Probabilidad de salida TensorBuffer.
     */
    private final TensorBuffer probabilityImageBuffer;
    /**
     * Procesador para aplicar postprocesamiento de la probabilidad de salida.
     */
    private final TensorProcessor probabilityProcessor;
    /**
     * Crea un clasificador
     *
     * @param activity the current activity
     * @throws IOException
     */
    public ClasificadorDeImagen(Activity activity) throws IOException {
        /*
         * El modelo cargado de TensorFlow Lite.
         */
        MappedByteBuffer classifierModel = FileUtil.loadMappedFile(activity,
                "mobilenet_v1_1.0_224_quant.tflite");
        // Carga etiquetas desde el archivo de etiquetas.
        labels = FileUtil.loadLabels(activity, "labels_mobilenet_quant_v1_224.txt");
        tensorClassifier = new Interpreter(classifierModel, null);
        //Lee el tipo y la forma de los tensores de entrada y salida, respectivamente
        int imageTensorIndex = 0; // input
        int probabilityTensorIndex = 0;// output
        int[] inputImageShape = tensorClassifier.getInputTensor(imageTensorIndex).shape();
        DataType inputDataType = tensorClassifier.getInputTensor(imageTensorIndex).dataType();
        int[] outputImageShape = tensorClassifier.getOutputTensor(probabilityTensorIndex).shape();
        DataType outputDataType = tensorClassifier.getOutputTensor(probabilityTensorIndex).dataType();
        imageResizeX = inputImageShape[1];
        imageResizeY = inputImageShape[2];
        // Lee el tipo y la forma de los tensores de entrada y salida, respectivamente.
        // Crea el tensor de entrada.
        inputImageBuffer = new TensorImage(inputDataType);
        // Crea el tensor de salida y su procesador.
        probabilityImageBuffer = TensorBuffer.createFixedSize(outputImageShape, outputDataType);
        // Crea el postprocesador para la probabilidad de salida.
        probabilityProcessor = new TensorProcessor.Builder().add(new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD))
                .build();
    }

    public List<Recognition> recognizeImage(final Bitmap bitmap, final int sensorOrientation) {
        List<Recognition> recognitions = new ArrayList<>();
        inputImageBuffer = loadImage(bitmap, sensorOrientation);
        tensorClassifier.run(inputImageBuffer.getBuffer(), probabilityImageBuffer.getBuffer().rewind());
        // Obtiene el mapa de etiqueta y probabilidad.
        Map<String, Float> labelledProbability = new TensorLabel(labels,
                probabilityProcessor.process(probabilityImageBuffer)).getMapWithFloatValue();
        for (Map.Entry<String, Float> entry : labelledProbability.entrySet()) {
            recognitions.add(new Recognition(entry.getKey(), entry.getValue()));
        }
        // Encuentre las mejores clasificaciones ordenando las predicciones según la confianza
        Collections.sort(recognitions);
        // devolviendo las 5 mejores predicciones
        return recognitions.subList(0, MAX_SIZE);
    }

    private TensorImage loadImage(Bitmap bitmap, int sensorOrientation) {
        // Carga un mapa de bits en una TensorImage.
        inputImageBuffer.load(bitmap);
        int noOfRotations = sensorOrientation / 90;
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        // Crea un procesador para TensorImage.
        // aquí se aplican los pasos previos al procesamiento
        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                .add(new ResizeOp(imageResizeX, imageResizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                .add(new Rot90Op(noOfRotations))
                .add(new NormalizeOp(IMAGE_MEAN, IMAGE_STD))
                .build();
        return imageProcessor.process(inputImageBuffer);
    }
    /**
     * Un resultado inmutable devuelto por un clasificador que describe lo que se reconoció.
     */
    public class Recognition implements Comparable {
        /**
         * Nombre para mostrar para el reconocimiento.
         */
        private String name;
        /**
         * Un puntaje clasificable de qué tan bueno es el reconocimiento en relación con
         * los demás. Cuanto más alto debería ser mejor.
         */
        private float confidence;
        public Recognition() {
        }
        public Recognition(String name, float confidence) {
            this.name = name;
            this.confidence = confidence;
        }
        public String getName() {
            return name;
        }
        public void setName(String name) {
            this.name = name;
        }
        public float getConfidence() {
            return confidence;
        }
        public void setConfidence(float confidence) {
            this.confidence = confidence;
        }
        @Override
        public String toString() {
            return "Recognition{" +
                    "name='" + name + '\'' +
                    ", confidence=" + confidence +
                    '}';
        }
        @Override
        public int compareTo(Object o) {
            return Float.compare(((Recognition) o).confidence, this.confidence);
        }
    }
}
