#include <Arduino.h>
#include <TensorFlowLite.h>

#include "samples_one_per_class.h" // has healthy_64x64x3[] and unhealthy_64x64x3[] (int8_t)

// Replace this include with the file you generated via `xxd -i`.
// Make sure the symbol names below match your generated names.
#include "model_data.h"
// extern const unsigned char model_data_[];
// extern const unsigned int model_data_len;

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;

  // Tune if you get allocation errors (watch serial logs). Start larger, then trim.
  constexpr int kTensorArenaSize = 200 * 1024;
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
}

// Choose which static sample to feed
#define USE_HEALTHY_SAMPLE 1

void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load model from compiled byte array
  model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
      "Model schema %d != supported %d",
      model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Op resolver: include kernels commonly needed by MobileNet-like models
  // Increase the template arg if you add more ops.
  static tflite::MicroMutableOpResolver<7> micro_op_resolver;
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddMul();
  micro_op_resolver.AddLogistic();
  // If your output is float, also add:
  micro_op_resolver.AddDequantize();    // harmless if unused; keeps flexibility

  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);

  // Expecting INT8 NHWC [1, 64, 64, 3]
  Serial.print("Input type: ");
  Serial.println(input->type);
  Serial.print("Input dims: ");
  for (int i = 0; i < input->dims->size; ++i) {
    Serial.print(input->dims->data[i]); Serial.print(i < input->dims->size-1 ? "x" : "\n");
  }

  if (input->type != kTfLiteInt8) {
    TF_LITE_REPORT_ERROR(error_reporter, "Model input is not INT8. Adjust code or re-export model.");
    return;
  }

  // Copy ONE static image (already quantized int8, NHWC 64x64x3) into the input tensor
#if USE_HEALTHY_SAMPLE
  const int8_t* sample = healthy_64x64x3;
  Serial.println("Feeding: healthy_64x64x3");
#else
  const int8_t* sample = unhealthy_64x64x3;
  Serial.println("Feeding: unhealthy_64x64x3");
#endif

  const size_t expected_bytes = 64 * 64 * 3; // elements (int8_t -> 1 byte each)
  if (input->bytes < expected_bytes) {
    TF_LITE_REPORT_ERROR(error_reporter, "Input tensor smaller than expected.");
    return;
  }
  memcpy(input->data.int8, sample, expected_bytes);

  Serial.println("Setup done.");
}

void loop() {
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    delay(2000);
    return;
  }

  TfLiteTensor* output = interpreter->output(0);

  // Binary classifier head options:
  //  - If Dense(1, sigmoid): output either FLOAT32 (prob) or INT8 (quantized prob).
  //  - If Dense(2, softmax): two logits/probs (class 0/1).
  const int out_dims = output->dims->size;
  const int out_count = output->dims->data[out_dims - 1];

  float prob_healthy = -1.f, prob_unhealthy = -1.f;

  if (out_count == 1) {
    // single logit/prob for "unhealthy" (or whichever you trained).
    if (output->type == kTfLiteInt8) {
      // Dequantize back to float probability
      const float scale = output->params.scale;
      const int zp = output->params.zero_point;
      int8_t q = output->data.int8[0];
      float p = scale * (static_cast<int>(q) - zp); // ~ [0,1]
      prob_unhealthy = p;            // adjust label meaning if needed
      prob_healthy   = 1.0f - p;
    } else if (output->type == kTfLiteFloat32) {
      float p = output->data.f[0];
      prob_unhealthy = p;
      prob_healthy   = 1.0f - p;
    } else {
      Serial.println("Unsupported output type");
    }
  } else if (out_count == 2) {
    // two-class output (softmax)
    if (output->type == kTfLiteInt8) {
      const float scale = output->params.scale;
      const int zp = output->params.zero_point;
      int8_t q0 = output->data.int8[0];
      int8_t q1 = output->data.int8[1];
      float p0 = scale * (static_cast<int>(q0) - zp);
      float p1 = scale * (static_cast<int>(q1) - zp);
      // If model outputs softmax, p0+p1≈1 and already probabilities.
      prob_healthy   = p0;  // check class index mapping in your training!
      prob_unhealthy = p1;
    } else if (output->type == kTfLiteFloat32) {
      prob_healthy   = output->data.f[0];
      prob_unhealthy = output->data.f[1];
    } else {
      Serial.println("Unsupported output type");
    }
  } else {
    Serial.print("Unexpected output size: ");
    Serial.println(out_count);
  }

  Serial.print("Healthy prob:   "); Serial.println(prob_healthy, 4);
  Serial.print("Unhealthy prob: "); Serial.println(prob_unhealthy, 4);

  delay(2000);
}
