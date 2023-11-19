# Enhancing-Digital-Reading-experience-through-dynamic-bounding-box-text-unblurring


## Introduction:


In today's world, many of us have shifted from reading physical books to digital versions. While this change offers convenience, it can also bring new problems like getting distracted and losing track of where we are in a text. Imagine you're reading an e-book, and a notification pops up, or you simply glance away from the screen. When you return, it's easy to lose your place or feel disoriented within the text.

To address these challenges, we've developed an exciting solution. We've harnessed technology, specifically bounding box technology and a tool called PyTesseract, to improve your digital reading experience. Instead of showing you the entire text all at once, we've come up with a unique method called "dynamic text unblurring." This approach considers how quickly you read and progressively reveals the text to you, one line at a time.

## Features:
The dynamic text unblurring system is designed with several features to enhance the digital reading experience. Key features include:

### Dynamic Text Unblurring:
  The system employs bounding box technology and PyTesseract to identify individual lines of text within e-books.
  Text is progressively unblurred based on the average time it takes for a reader to absorb a single line, preventing distractions and maintaining focus.

### Text-to-Speech Integration:
   The dynamically unblurred text can be automatically converted to speech using a Text-to-Speech (TTS) engine.
   This feature accommodates users who prefer auditory information or those with visual impairments.

   
### Enhanced Reading Comprehension:
   he progressive unblurring approach prevents users from losing their place in the text, contributing to improved reading comprehension.
    
## Requirements
### Hardware Requirements
    Computer or Laptop:
    Speakers or Headphones (Optional):
### Softare Requirements

#### Operating System:  
Windows, macOS, or Linux.
#### Development Environment:  
Visual Studio Code, PyCharm, or others.
#### Programming Language: 
Python, and relevant libraries and frameworks.
#### Image Processing Libraries: 
Integration of image processing libraries like OpenCV for tasks such as blurring, unblurring, and image manipulation.
#### Text Recognition Library: 
Integration of OCR (Optical Character Recognition) libraries such as Tesseract to accurately recognize and extract text from images.
#### Text-to-Speech (TTS) Engine:
Integration with a TTS engine, such as pyttsx3 or other suitable alternatives, to convert dynamically unblurred text into speech.


## Program
```python

import cv2
import pytesseract
import numpy as np
import time
import pyttsx3

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def elapsed_time_function():
    start_time = None
    end_time = None

    total_lines = 20  # Total number of lines

    while True:
        key = cv2.waitKey(1) & 0xFF

        # Start the timer when "q" is pressed
        if key == ord('q') and start_time is None:
            start_time = time.time()
            print("Timer started")

        # Stop the timer when "q" is pressed again
        elif key == ord('q') and start_time is not None and end_time is None:
            end_time = time.time()
            print("Timer stopped")

        if end_time is not None:
            break

    # Calculate the time elapsed
    elapsed_time = end_time - start_time

    # Calculate time per line
    time_per_line = elapsed_time / total_lines
    time_per_line = int(time_per_line * 1000)  # Convert to milliseconds

    print(f"Time per line: {time_per_line} milliseconds per line")
    return time_per_line

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load the image
image_path = 'largepreview.png'  # Replace with your image file path
image = cv2.imread(image_path)

# Display the image
cv2.imshow("Image", image)

# Call the elapsed_time_function to get the time_per_line
time_per_line = elapsed_time_function()

# Clean up
cv2.destroyAllWindows()

# Load the image again for text extraction
img = cv2.imread(image_path)

# Convert the image to grayscale for better text extraction
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Use Pytesseract to extract the text and its bounding boxes
detections = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

# Initialize variables to group bounding boxes by lines
line_boxes = []
current_line = []

for i in range(len(detections['text'])):
    x, y, w, h = detections['left'][i], detections['top'][i], detections['width'][i], detections['height'][i]
    text = detections['text'][i]

    # Check if the text is non-empty and not a space (to ignore empty lines)
    if text.strip():
        if not current_line:
            current_line.append((x, y, x + w, y + h, text))
        else:
            prev_x2 = current_line[-1][2]
            if x > prev_x2:  # Check if the current box is to the right of the previous box
                current_line.append((x, y, x + w, y + h, text))
            else:
                line_boxes.append(current_line)
                current_line = [(x, y, x + w, y + h, text)]

# Append the last line
if current_line:
    line_boxes.append(current_line)

# Initialize a strong blur image
kernel_size = (15, 15)  # Adjust the kernel size as needed
strong_blur_img = cv2.blur(image, kernel_size)

# Create a copy of the strong blur image for unblurring
current_img = strong_blur_img.copy()

# Iterate through each line and progressively unblur one line at a time
for line in line_boxes:
    for box in line:
        min_x, min_y, max_x, max_y, text = box
        current_img[min_y:max_y, min_x:max_x] = image[min_y:max_y, min_x:max_x]

    # Display the current state of the image with one line unblurred
    cv2.imshow("Unblurred Image", current_img)
    cv2.waitKey(time_per_line)  # Adjust the delay as needed (1500 milliseconds = 1.5 seconds)

    # Restore the cumulative image without the current line
    for box in line:
        min_x, min_y, max_x, max_y, _ = box
        current_img[min_y:max_y, min_x:max_x] = strong_blur_img[min_y:max_y, min_x:max_x]

    # Perform text-to-speech conversion
    spoken_text = ' '.join([box[4] for box in line])

    # If text-to-speech conversion is successful, speak the text
    if spoken_text.strip():
        engine.say(spoken_text)
        engine.runAndWait()

# Close the OpenCV window
cv2.destroyAllWindows()


```


## Output

### User average reading speed of single line

![Screenshot 2023-11-20 023458](https://github.com/Dineshkumar200/Enhancing-Digital-Reading-experience-through-dynamic-bounding-box-text-unblurring/assets/75235789/e69e6bab-e8fa-4cde-8019-ed18fcaeac18)


### Dynamic Text Unblurring
After Calculating the user speed time of the single line the entire will be converted into blur after that based on average user reading speed it will unblur individual line dynamically at the same time the unblurred text will be converted into text to speech. After that this line automatically blur and the next line will be unblur


![Screenshot (11)](https://github.com/Dineshkumar200/Enhancing-Digital-Reading-experience-through-dynamic-bounding-box-text-unblurring/assets/75235789/b8d7be3f-04d2-4d98-a312-5c77f53b8747)

## Result


This project introduces a novel approach to enhance the digital reading experience through dynamic bounding box text unblurring. The methodology developed utilizes OCR technology and bounding boxes to isolate individual lines of text within e-books and digital content. By dynamically unblurring text based on the reader’s speed, this approach addresses the common issues of reader distraction and the loss of one's place within a text.
The testing and evaluation of the system demonstrate its potential to significantly improve reading comprehension and reduce distractions, ultimately providing a more user-friendly and immersive digital reading experience. The interactive features allow readers to tailor their reading experience to their preferences.As e-books and digital content continue to gain popularity, this project offers a practical and effective solution to enhance the quality of digital reading..

