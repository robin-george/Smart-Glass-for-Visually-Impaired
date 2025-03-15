# Smart-Glass-for-Visually-Impaired
A wearable assistive device designed to enhance navigation for visually impaired users. Built using **Raspberry Pi 5** and **Camera Module 3**, this project provides real-time **obstacle detection**, **object recognition**, **reading assistance**, and **auditory feedback** for safer navigation. 

# **Features**
- **Real-Time Obstacle Detection:** Uses **edge detection** to identify obstacles in the user's path.  
- **Object Recognition:** Implements **YOLO (You Only Look Once)** for detecting and classifying objects.  
- **Reading Assistance:** Uses **Tesseract OCR (pytesseract)** to read printed text aloud.  
- **Auditory Feedback:** Converts detected information into speech using **pyttsx3**.

# **Hardware Components**
- **Raspberry Pi 5**  
- **Raspberry Pi Camera Module 3**    
- **Headphone for Audio Output**

# **Software & Libraries Used**  
- **Python** (main programming language)  
- **OpenCV** (for edge detection)  
- **YOLOv8** (for object recognition)  
- **Pytesseract** (OCR for text reading)  
- **Pyttsx3** (text-to-speech conversion)  

# **How It Works**  
1. **Obstacle Detection:** Uses **OpenCV edge detection** to analyze the surroundings.  
2. **Object Recognition:** Runs **YOLOv8** on the Raspberry Pi for real-time object classification.  
3. **Text Recognition:** Captures images of text and converts them into speech using **Tesseract OCR**.  
4. **Audio Output:** The processed information is communicated to the user via **text-to-speech (TTS)**.  

## License  
This project is licensed under the **MIT License**. 
