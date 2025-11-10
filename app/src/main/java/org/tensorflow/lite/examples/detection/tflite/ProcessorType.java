package org.tensorflow.lite.examples.detection.tflite;

/**
 * Enum representing different processor types for TensorFlow Lite inference.
 */
public enum ProcessorType {
    CPU(0),
    NPU(1),  // NNAPI
    GPU(2);

    private final int value;

    ProcessorType(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }

    /**
     * Convert integer value to ProcessorType enum
     * @param value integer value (0=CPU, 1=NPU, 2=GPU)
     * @return corresponding ProcessorType
     */
    public static ProcessorType fromValue(int value) {
        switch (value) {
            case 0: return CPU;
            case 1: return NPU;
            case 2: return GPU;
            default: return CPU; // Default to CPU
        }
    }
}
