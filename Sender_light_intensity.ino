#include <Arduino_APDS9960.h>
#include <ArduinoBLE.h>

// BLE service and characteristic
BLEService myService("fff0");
BLEIntCharacteristic myCharacteristic("fff1", BLERead | BLEBroadcast);

void setup() {
  // Initialize Serial Monitor
  Serial.begin(9600);
  while (!Serial);

  // Initialize APDS9960 Sensor
  if (!APDS.begin()) {
    Serial.println("Error initializing APDS9960 sensor.");
    while (1); // Halt if the sensor initialization fails
  }

  // Initialize BLE
  if (!BLE.begin()) {
    Serial.println("Failed to initialize BLE!");
    while (1);
  }

  // Add BLE characteristic and service
  myService.addCharacteristic(myCharacteristic);
  BLE.addService(myService);

  // Set BLE local name and start advertising
  BLE.setLocalName("Sender_2");
  BLE.advertise();

  Serial.println("BLE advertising and sensor setup done.");
}

void loop() {
  // Poll BLE for events
  BLE.poll();

  // Check if a color reading is available
  while (!APDS.colorAvailable()) {
    delay(5);
  }

  int r, g, b;
  APDS.readColor(r, g, b); // Read RGB values from the APDS9960 sensor

  // Send the red intensity value via BLE
  myCharacteristic.writeValue(r);

  // Print the red value for debugging
  Serial.print("Red intensity sent: ");
  Serial.println(r);

  // Delay for stability
  delay(500);
}
