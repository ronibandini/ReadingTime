/*  Reading Time
 *  Machine Learning, Arduino BLE 33 Sense and Edge Impulse
 *  Roni Bandini @RoniBandini, Buenos Aires, March 2021
 *  tags: hojas (pages), silencio (silence)
 */

#define EIDSP_QUANTIZE_FILTERBANK   0


#include <PDM.h>
#include <bandini-project-1_inference.h>
#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128  
#define SCREEN_HEIGHT 32  
#define OLED_RESET     4  
#define SCREEN_ADDRESS 0x3C  
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

/** Audio buffers, pointers and selectors */
typedef struct {
    int16_t *buffer;
    uint8_t buf_ready;
    uint32_t buf_count;
    uint32_t n_samples;
} inference_t;

static inference_t inference;
static signed short sampleBuffer[2048];
static bool debug_nn = false;  

#define imageWidthLogo 128
#define imageHeightLogo 32

const unsigned char logo [] PROGMEM = {
0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe, 0x1f, 0xfe, 0x3f, 0xff, 
0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfc, 0x03, 0xe0, 0x1f, 0xff, 
0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfd, 0xf1, 0xc7, 0x9f, 0xff, 
0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfd, 0xfc, 0x9f, 0xdf, 0xff, 
0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe1, 0xfe, 0x3f, 0xc3, 0xff, 
0xff, 0x83, 0x83, 0x0c, 0x0e, 0x66, 0x70, 0xe0, 0x44, 0x71, 0x87, 0xe5, 0xff, 0x7f, 0xd3, 0xff, 
0xff, 0x81, 0x83, 0x0c, 0x06, 0x66, 0x60, 0x60, 0x44, 0x61, 0x87, 0xed, 0xff, 0x7f, 0x93, 0xff, 
0xff, 0x89, 0x8f, 0x0c, 0x46, 0x62, 0x62, 0x71, 0xc4, 0x21, 0x9f, 0xed, 0xff, 0x7f, 0x93, 0xff, 
0xff, 0x89, 0x8e, 0x0c, 0x46, 0x62, 0x66, 0x71, 0xc4, 0x21, 0x9f, 0xed, 0xff, 0x7f, 0x93, 0xff, 
0xff, 0x89, 0x8e, 0x4c, 0x46, 0x62, 0x66, 0x71, 0xc4, 0x21, 0x9f, 0xed, 0xff, 0x7f, 0x93, 0xff, 
0xff, 0x89, 0x8e, 0x4c, 0x46, 0x60, 0x66, 0x71, 0xc4, 0x21, 0x9f, 0xed, 0xff, 0x7f, 0x93, 0xff, 
0xff, 0x81, 0x86, 0x4c, 0x46, 0x60, 0x67, 0xf1, 0xc4, 0x01, 0x87, 0xed, 0xff, 0x7f, 0x93, 0xff, 
0xff, 0x83, 0x86, 0x64, 0x46, 0x60, 0x64, 0x71, 0xc4, 0x01, 0x87, 0xed, 0xff, 0x7f, 0x93, 0xff, 
0xff, 0x89, 0x8e, 0x64, 0x46, 0x60, 0x64, 0x71, 0xc4, 0x81, 0x9f, 0xed, 0xff, 0x7f, 0x93, 0xff, 
0xff, 0x89, 0x8e, 0x64, 0x46, 0x60, 0x66, 0x71, 0xc4, 0x81, 0x9f, 0xed, 0xff, 0x7f, 0x93, 0xff, 
0xff, 0x89, 0x8e, 0x04, 0x46, 0x64, 0x66, 0x71, 0xc4, 0x81, 0x9f, 0xed, 0xff, 0x7f, 0x93, 0xff, 
0xff, 0x89, 0x8e, 0x04, 0x46, 0x64, 0x66, 0x71, 0xc4, 0x99, 0x9f, 0xec, 0x3f, 0x7e, 0x13, 0xff, 
0xff, 0x89, 0x8c, 0x64, 0x46, 0x66, 0x62, 0x71, 0xc4, 0x99, 0x9f, 0xec, 0x07, 0x70, 0x13, 0xff, 
0xff, 0x89, 0x80, 0x64, 0x06, 0x66, 0x60, 0x71, 0xc4, 0x99, 0x83, 0xef, 0xf0, 0x47, 0xf3, 0xff, 
0xff, 0x89, 0x80, 0x64, 0x0e, 0x66, 0x71, 0x71, 0xc4, 0xd9, 0x83, 0xef, 0xfc, 0x1d, 0xf3, 0xff, 
0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe0, 0x00, 0x00, 0x03, 0xff, 
0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe3, 0xfc, 0x1f, 0xe3, 0xff, 
0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
};

int pages=0;
int pagesCounter=1;
int myPages=0;

unsigned long StartTime = millis();

void setup()
{

    Serial.begin(115200);

    delay(3000);  //while(!Serial);

    Serial.println("Reading Time");
    Serial.println("March 2021 @RoniBandini");
    Serial.println("-----------------------");

    if(!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
        Serial.println(F("SSD1306 allocation failed"));
        for(;;); // Don't proceed, loop forever
    }

  display.display();
  delay(2000); 
  display.clearDisplay();   
  display.drawBitmap(0, 0,  logo, imageWidthLogo, imageHeightLogo, 1); 
  display.display();   
  delay(3000);

  display.clearDisplay();
  display.setTextSize(1);            
  display.setTextColor(SSD1306_WHITE); 
  display.println("Reading time");
  display.println("Machine Learning"); 
  display.println("by @RoniBandini"); 
  display.display();
  delay(3000); 

  // Read potentiometer to determine book pages
  
  uint32_t period = 1 * 15000L;       // 15 seconds

  for( uint32_t tStart = millis();  (millis()-tStart) < period;  ){       
  
       int potValue = analogRead(A0);
       myPages  = map(potValue, 0, 1023, 80, 500);
       
       display.clearDisplay();
       display.setCursor(0,0);
       display.setTextSize(2); 
       display.println("Pages");        
       display.println(myPages); 
       
       Serial.println("Pages");
       Serial.println(myPages);
       
       display.display();
       delay(2000);
   }
  
  display.println("Book pages configured: "); 
  Serial.print(myPages); 
  pages=myPages; 
       
  ei_printf("Inferencing settings:\n");
  ei_printf("\tInterval: %.2f ms.\n", (float)EI_CLASSIFIER_INTERVAL_MS);
  ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
  ei_printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
  ei_printf("\tNo. of classes: %d\n", sizeof(ei_classifier_inferencing_categories) / sizeof(ei_classifier_inferencing_categories[0]));

  if (microphone_inference_start(EI_CLASSIFIER_RAW_SAMPLE_COUNT) == false) {
        ei_printf("ERR: Failed to setup audio sampling\r\n");
        return;
  }

  display.clearDisplay();
  display.setCursor(0,0);
  display.setTextSize(2); 
  display.println("Start");  
  display.println("Reading");     
  display.display();  
    
}


void loop()
{
    ei_printf("Inferencing in some seconds...\n");
    delay(400);

    ei_printf("Listening...\n");

    bool m = microphone_inference_record();
    if (!m) {
        ei_printf("ERR: Failed to record audio...\n");
        return;
    }

    ei_printf("Listening done\n");

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    signal.get_data = &microphone_audio_signal_get_data;
    ei_impulse_result_t result = { 0 };

    EI_IMPULSE_ERROR r = run_classifier(&signal, &result, debug_nn);
    if (r != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", r);
        return;
    }

    // print predictions
    ei_printf("Predictions ");
    ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
        result.timing.dsp, result.timing.classification, result.timing.anomaly);
    ei_printf(": \n");


    display.clearDisplay();
    display.setCursor(0,0);
    display.setTextSize(2);      
    
    float hojas = 0;
    float silencio = 0;
        
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        ei_printf("    %s: %.5f\n", result.classification[ix].label, result.classification[ix].value);
        
         
          if (result.classification[ix].label=="hojas"){
            hojas=result.classification[ix].value;
          }
          
          if (result.classification[ix].label=="silencio"){
            silencio=result.classification[ix].value;
          }

       
    }

    if (hojas>0.80){
      
      display.println("Page");  
      display.println("turned");     
      //display.println(String(hojas*100)+"%");
      display.display();  
      
      // add 2 pages
      pagesCounter=pagesCounter+2;

      // calculate remaining time
      unsigned long CurrentTime = millis();
      unsigned long ElapsedTime = CurrentTime - StartTime;

      Serial.println("ElapsedTime:");
      Serial.print(ElapsedTime);
      Serial.println("PagesCounter:");
      Serial.print(pagesCounter);
      
      int remainingMinutes=(ElapsedTime/pagesCounter)*(pages-pagesCounter)/1000/60;

      Serial.println("Remaining mins to finish book:");
      Serial.print(remainingMinutes);

      delay(1000);
      display.clearDisplay();
      display.setCursor(0,0);
      display.println("Remaining");      
      display.println(String(remainingMinutes)+" min");
      display.display(); 

    }
    else
    {
      display.println("Reading");
      display.println("Silence");
      //Use this to print score display.println(String(silencio*100)+"%");  
      //Use this to print current page display.println(String(pagesCounter)+"/"+pages);        
      display.display();
      
      }                
    
    
#if EI_CLASSIFIER_HAS_ANOMALY == 1
    ei_printf("    anomaly score: %.3f\n", result.anomaly);
#endif
}

/**
 * @brief      Printf function uses vsnprintf and output using Arduino Serial
 *
 * @param[in]  format     Variable argument list
 */
void ei_printf(const char *format, ...) {
    static char print_buf[1024] = { 0 };

    va_list args;
    va_start(args, format);
    int r = vsnprintf(print_buf, sizeof(print_buf), format, args);
    va_end(args);

    if (r > 0) {
        Serial.write(print_buf);
    }
}

/**
 * @brief      PDM buffer full callback
 *             Get data and call audio thread callback
 */
static void pdm_data_ready_inference_callback(void)
{
    int bytesAvailable = PDM.available();

    // read into the sample buffer
    int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);

    if (inference.buf_ready == 0) {
        for(int i = 0; i < bytesRead>>1; i++) {
            inference.buffer[inference.buf_count++] = sampleBuffer[i];

            if(inference.buf_count >= inference.n_samples) {
                inference.buf_count = 0;
                inference.buf_ready = 1;
                break;
            }
        }
    }
}


static bool microphone_inference_start(uint32_t n_samples)
{
    inference.buffer = (int16_t *)malloc(n_samples * sizeof(int16_t));

    if(inference.buffer == NULL) {
        return false;
    }

    inference.buf_count  = 0;
    inference.n_samples  = n_samples;
    inference.buf_ready  = 0;

    // configure the data receive callback
    PDM.onReceive(&pdm_data_ready_inference_callback);

    // optionally set the gain, defaults to 20
    PDM.setGain(80);
    PDM.setBufferSize(4096);

    // initialize PDM with:
    // - one channel (mono mode)
    // - a 16 kHz sample rate
    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {
        ei_printf("Failed to start PDM!");
        microphone_inference_end();

        return false;
    }

    return true;
}

/**
 * @brief      Wait on new data
 *
 * @return     True when finished
 */
static bool microphone_inference_record(void)
{
    inference.buf_ready = 0;
    inference.buf_count = 0;

    while(inference.buf_ready == 0) {
        delay(10);
    }

    return true;
}

/**
 * Get raw audio signal data
 */
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    numpy::int16_to_float(&inference.buffer[offset], out_ptr, length);

    return 0;
}

/**
 * @brief      Stop PDM and release buffers
 */
static void microphone_inference_end(void)
{
    PDM.end();
    free(inference.buffer);
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif
