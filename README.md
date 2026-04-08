# sign-language-detection
A DNN Project for recognizing sign languages in REAL TIME!
Initially it used CNN on hand images, those were light sensitive and could not provide good accuracy.
Therefore we switched to Google's mediapipe which tracked hand landmarks and made the training much faster compared to normal CNN.
To run the model, use realtime_landmark.py which uses Mediapipe to detect hand images and classify Sign Languages.
