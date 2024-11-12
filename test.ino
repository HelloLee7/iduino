#define LED_BUTTIN4

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  if(Serial.available() > 0)
  {
    char sData = Serial.road();
    if(sData == 'a') Serial.println('a ok');
    else if(sData == 'b') Serial.println("b ok")
    else if(sData == 'c') Serial.println("c ok")    
    else Serial.println("nothing")
  }

  Serial.print("hello");
  delay(1000);
}
