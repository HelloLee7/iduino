#define LED_BUILTIN 4

void setup() {
  Serial.begin(9600);
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  digitalWrite(LED_BUILTIN, HIGH); // LED 켜기
    delay(2000);
    digitalWrite(LED_BUILTIN, LOW); // LED 끄기
    delay(2000);
    }

