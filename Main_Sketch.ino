#include <Arduino_LSM9DS1.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <TinyMLShield.h>
#include <ArduinoBLE.h>


// NN parameters, set these yourself! 
#define LEARNING_RATE 0.1    // The learning rate used to train your network
#define EPOCH 28             // The maximum number of epochs 
#define DATA_TYPE_FLOAT      // The data type used: Set this to DATA_TYPE_DOUBLE for higher precision

extern const int first_layer_input_cnt;
extern const int classes_cnt;
int light_memory;
// You define your network in NN_def
// Right now, the network consists of three layers: 
// 1. An input layer with the size of your input as defined in the variable first_layer_input_cnt in cnn_data.h 
// 2. A hidden layer with 20 nodes
// 3. An output layer with as many classes as you defined in the variable classes_cnt in cnn_data.h 
static const unsigned int NN_def[] = {first_layer_input_cnt, 20, classes_cnt};

#include "data.h"       // The data, labels and the sizes of all objects are stored here 
#include "NN_functions.h"   // All NN functions are stored here 

int iter_cnt = 0;           // This keeps track of the number of epochs you've trained on the Arduino
#define DEBUG 0             // This prints the weights of your network in case you want to do debugging (set to 1 if you want to see that)

// Global variables for IMU data and normalization
float gx, gy, gz;
float gx_min = FLT_MAX, gx_max = -FLT_MAX;
float gy_min = FLT_MAX, gy_max = -FLT_MAX;
float gz_min = FLT_MAX, gz_max = -FLT_MAX;

const float accelerationThreshold = 2.5;
const int numSamples = 119;
int samplesRead = numSamples;

#define FLATTEN_LENGTH 357  // 3 * 119

// Calibration function to find min and max values
void calibrateIMU() {
    Serial.println("Calibrating IMU. Keep the device still...");
    
    for (int i = 0; i < 500; i++) {  
        if (IMU.gyroscopeAvailable()) {
            IMU.readGyroscope(gx, gy, gz);
            
            gx_min = min(gx_min, gx);
            gx_max = max(gx_max, gx);
            gy_min = min(gy_min, gy);
            gy_max = max(gy_max, gy);
            gz_min = min(gz_min, gz);
            gz_max = max(gz_max, gz);
            
            delay(10);
        }
    }
    
    Serial.println("Calibration complete.");
    Serial.print("Gyro X range: "); Serial.print(gx_min); Serial.print(" to "); Serial.println(gx_max);
    Serial.print("Gyro Y range: "); Serial.print(gy_min); Serial.print(" to "); Serial.println(gy_max);
    Serial.print("Gyro Z range: "); Serial.print(gz_min); Serial.print(" to "); Serial.println(gz_max);
}

// Normalize the IMU data
float normalizeValue(float value, float min_val, float max_val) {
    return (value - min_val) / (max_val - min_val) * 2.0 - 1.0;
}

// Perform inference on current IMU data
int performInference(float* confidence = nullptr) {
    float flattenedInput[FLATTEN_LENGTH];
    int label = -1;
    unsigned long startWaitTime = millis();
    float aX, aY, aZ;

    // Wait for significant motion
    while (samplesRead == numSamples) {
        if (IMU.accelerationAvailable()) {
            // Read the acceleration data
            IMU.readAcceleration(aX, aY, aZ);

            // Sum up the absolutes
            float aSum = fabs(aX) + fabs(aY) + fabs(aZ);

            // Check if it's above the threshold
            if (aSum >= accelerationThreshold) {
                // Reset the sample read count
                samplesRead = 0;
                break;
            }
        }

        // Timeout to prevent infinite loop
        if (millis() - startWaitTime > 6000) { // Timeout after 5 seconds
            Serial.println("Timeout: No motion detected.");
            
            return -1;
        }
    }

    // Collect gyroscope samples
    int time_step = 0;
    while (samplesRead < numSamples) {
        if (IMU.gyroscopeAvailable()) {
            IMU.readGyroscope(gx, gy, gz);
            
            // Normalize the input
            float normalized_gx = normalizeValue(gx, gx_min, gx_max);
            float normalized_gy = normalizeValue(gy, gy_min, gy_max);
            float normalized_gz = normalizeValue(gz, gz_min, gz_max);
            
            // Store in flattened array
            flattenedInput[time_step] = normalized_gx;
            flattenedInput[time_step + 119] = normalized_gy;
            flattenedInput[time_step + 119 + 119] = normalized_gz;
            
            time_step++;
            samplesRead++;


            // When we have collected all samples
            if (samplesRead == numSamples) {
                // Copy flattened input to the global input array
                for (int i = 0; i < FLATTEN_LENGTH; i++) {
                    input[i] = flattenedInput[i];
                }

                // Perform forward propagation
                forwardProp();
                
                // Get the predicted class
                label = getPrediction(confidence);
                break;
            }

            delay(10);  // Adjust based on original data collection rate
        }
    }

    // Return the predicted class
    return label;
}

// This function contains your training loop 
void do_training() {
    // Print the weights if you want to debug 
    #if DEBUG      
        Serial.println("Now Training");
        PRINT_WEIGHTS();
    #endif

    // Print the epoch number 
    Serial.print("Epoch count (training count): ");
    Serial.print(++iter_cnt);
    Serial.println();

    // Reordering the index for more randomness and faster learning
    shuffleIndx();
  
    // Starting forward + Backward propagation
    for (int j = 0; j < numTrainData; j++) {
        generateTrainVectors(j);  
        forwardProp();
        backwardProp();
    }

    Serial.println("Accuracy after local training:");
    printAccuracy();
}

void setup() {
    // Initialize random seed 
    srand(0); 
  
    Serial.begin(9600); 
    delay(5000);
    while (!Serial); 

    // Initialize IMU
    if (!IMU.begin()) {
        Serial.println("Failed to initialize IMU!");
        while (1);
    }
  
    // Initialize pins as outputs
    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);

    // Calibrate IMU
    calibrateIMU();

    // Initialize the TinyML Shield 
    initializeShield();

    // Calculate how many weights and biases we're training on the device
    int weights_bias_cnt = calcTotalWeightsBias(); 

    Serial.print("The total number of weights and bias used for on-device training on Arduino: ");
    Serial.println(weights_bias_cnt);

    // Allocate common weight vector, and pass to setupNN, setupBLE
    DATA_TYPE* WeightBiasPtr = (DATA_TYPE*) calloc(weights_bias_cnt, sizeof(DATA_TYPE));

    setupNN(WeightBiasPtr);  // CREATES THE NETWORK BASED ON NN_def[]
    Serial.print("The accuracy before training: ");
    printAccuracy();
  
    Serial.println("Use the on-shield button to start and stop the loop code ");

    // Initialize BLE
    if (!BLE.begin()) {
      Serial.println("Failed to start BLE!");
      while (1);
    }

    Serial.println("BLE Central scanning for peripherals...");

    // Start scanning for peripherals
    BLE.scan();
}

void loop() {

  float confidence = 0.0;
  int prediction;
  // Train the network for specified number of epochs
  while (iter_cnt < EPOCH) {
      bool clicked = readShieldButton();
      if (clicked) {
          Serial.println("Button clicked, starting training");
          do_training(); // Local training 
      }
  }

  // receive light values from other Arduino
  // Check if a peripheral has been discovered
  int32_t redValue;  // Correct data type
  BLEDevice peripheral = BLE.available();

  // Look for a device with the name "Sender"
  if (peripheral && peripheral.hasLocalName() && peripheral.localName().startsWith("Sender")) {
    Serial.println("Discovered a peripheral!");
    Serial.println("-----------------------");

    // Print the peripheral's address and local name
    Serial.print("Address: ");
    Serial.println(peripheral.address());

    Serial.print("Local Name: ");
    Serial.println(peripheral.localName());

    // Stop scanning
    BLE.stopScan();

    unsigned long onWaitTime = millis();
    // Attempt to connect to the peripheral
    if (peripheral.connect()) {
      Serial.println("Connected to the peripheral!");

      // Discover peripheral's services
      if (peripheral.discoverAttributes()) {
        Serial.println("Attributes discovered.");

        // Look for the characteristic with UUID "fff1"
        BLECharacteristic redCharacteristic = peripheral.characteristic("fff1");

        if (redCharacteristic) {
          Serial.println("Found Red Intensity Characteristic.");

          // Continuously read red intensity values
          while (peripheral.connected()) {
            if (redCharacteristic.canRead()) {
              
            
      
              if (redCharacteristic.readValue(redValue)) {
                Serial.print("Red Intensity BL Loop: ");
                Serial.println(redValue);

                // Perform inference and print prediction
                prediction = performInference(&confidence);
                
                if (prediction != -1) {
                  Serial.print("Predicted Gesture: ");
                  Serial.print(prediction);
                  // Serial.print(" (Confidence: ");
                  // Serial.print(confidence * 100);
                  // Serial.println("%)");
                  samplesRead = numSamples;  // Reset for next inference

                  if (prediction == 0 ){
                    // turn the lights OFF
                    analogWrite(LEDR,255);
                    analogWrite(LEDG,255);
                    analogWrite(LEDB,255);
                    light_memory = 255;
                  }
                  else if (prediction ==1 ){
                    // turn the lights ON
                    analogWrite(LEDR,0);
                    analogWrite(LEDG,0);
                    analogWrite(LEDB,0);
                    light_memory = 0;
                    onWaitTime = millis();
                  }
                  else if (prediction ==2){
                    // dim the lights
                    if (light_memory==0){
                      for (int brightness = 0; brightness <= 127; brightness++) {
                      analogWrite(LEDR,brightness);
                      analogWrite(LEDG,brightness);
                      analogWrite(LEDB,brightness);
                      delay(10);
                    }
                    light_memory = 127;
                    }
                    else if (light_memory == 127){
                      for (int brightness = 127; brightness <= 255; brightness++) {
                      analogWrite(LEDR,brightness);
                      analogWrite(LEDG,brightness);
                      analogWrite(LEDB,brightness);
                      delay(10);
                    }
                    light_memory = 255;
                    }
                    
                  }
                  else if (prediction ==3){
                    // brightening the lights
                    if (light_memory == 255){
                      for (int brightness = 255; brightness >= 127; brightness--) {
                      analogWrite(LEDR,brightness);
                      analogWrite(LEDG,brightness);
                      analogWrite(LEDB,brightness);
                      onWaitTime = millis();
                      delay(10);

                    }
                    light_memory = 127;
                    }
                    else if (light_memory == 127){
                      for (int brightness = 127; brightness >= 0; brightness--) {
                      analogWrite(LEDR,brightness);
                      analogWrite(LEDG,brightness);
                      analogWrite(LEDB,brightness);
                      onWaitTime = millis();
                      delay(10);
                    }
                    light_memory = 0;
                    }

                  }
                    
                }
                
                // int Intensity = 255 - round(redValue);
                // analogWrite(LEDR, Intensity);
                // analogWrite(LEDG, Intensity);
                // analogWrite(LEDB, Intensity);

                if (redValue > 200 && (millis() - onWaitTime > 60000)) // 1-minute is the waiting period
                {
                  // turn the lights OFF
                    analogWrite(LEDR,255);
                    analogWrite(LEDG,255);
                    analogWrite(LEDB,255);
                    light_memory = 255;
                }

              } else {
                Serial.println("Failed to read Red Intensity value.");
              }
            }
            delay(3000); // Add a delay for stability
          }
        } else {
          Serial.println("Red Intensity Characteristic not found.");
        }
      } else {
        Serial.println("Failed to discover attributes.");
      }

      // Disconnect from the peripheral
      peripheral.disconnect();
      Serial.println("Disconnected from the peripheral.");
    } else {
      Serial.println("Failed to connect to the peripheral.");
    }

    // Restart scanning after disconnection
    BLE.scan();
  }
 ////////

}