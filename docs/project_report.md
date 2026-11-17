# PROJECT REPORT
## Pakistan Sign Language (PSL) Alphabet Recognizer

**Course**: AL-2002 Artificial Intelligence (Laboratory), Fall 2026  
**Institution**: FAST National University of Computer and Emerging Sciences, Karachi Campus  
**Department**: Department of Software Engineering  
**Student Name**: Areesha Ahmad  
**Roll Number**: 24K-3043 (BSE-5B)  
**Project Category**: Applied Machine Learning / Data Science (Solo Project)  
**Repository URL**: [https://github.com/arwsh21/urdu-sign-lang](https://github.com/arwsh21/urdu-sign-lang)  

---

## 1. Project Motivation

Sign language is the primary language used by deaf individuals to communicate. In Pakistan, the deaf community relies on Pakistan Sign Language (PSL). Most digital translation tools and accessibility research focus heavily on American Sign Language (ASL) or British Sign Language (BSL). Very few software tools exist for Pakistan Sign Language.

A major barrier for deaf individuals in Pakistan is communicating with hearing people who do not understand sign language. Traditional human interpreters are expensive and not always available. A computer vision program that translates PSL alphabet signs into text using an ordinary laptop camera provides a practical, low cost starting point for communication and learning.

Writing manual rules for recognizing hand signs (such as "if the index finger is pointing up, it is an Alif") fails in practice. Every person has a different hand size, holds their hand at a slightly different angle, and sits at a different distance from the camera. Variations in room lighting also alter the visual appearance of the hand. Machine learning solves this problem by discovering the geometric boundaries directly from recorded examples.

---

## 2. Project Objectives

The core objectives of this project are:
1. To collect a primary dataset of 31 static alphabet signs from Pakistan Sign Language using a standard laptop webcam.
2. To extract 3D joint landmarks of the hand without relying on heavy deep learning frameworks.
3. To design a geometric normalisation algorithm that makes hand representations invariant to screen position and camera distance.
4. To build an object-oriented classifier architecture comparing three machine learning models (K-Nearest Neighbors, Support Vector Machine, and Random Forest) against a baseline.
5. To implement an intelligent smoothing agent and word builder that filters camera noise and spells out words in both Roman Urdu and native conjoined Urdu text.
6. To conduct a 20-seed experiment evaluating accuracy, Macro F1 score, training time, and per-sample prediction speed.

---

## 3. Problem Formulation

### 3.1 PEAS Description
The system is modeled as an AI agent operating in an interactive environment.

* **Performance Measure**: High classification accuracy (greater than 90%), balanced Macro F1 score across all 31 letters, fast prediction time (under 1 millisecond per frame), and reliable word spelling without stuttering.
* **Environment**: A person sitting in an indoor room in front of a laptop webcam, showing static one-handed PSL alphabet shapes.
* **Actuators**: The user interface screen displaying the live camera video, tracked skeletal joints, detected letter name, confidence percentage, hold progress bar, and the formed word.
* **Sensors**: The laptop webcam capturing raw video frames, coupled with MediaPipe Hands extracting 21 three-dimensional landmark points.

### 3.2 Environment Characteristics
* **Partially Observable**: The camera only captures the 2D visual projection of the hand. Self-occlusions (where one finger hides behind another) cause partial visibility.
* **Stochastic**: Sensor noise, camera blur, and micro-movements in the hand introduce random variations between frames.
* **Dynamic**: The physical hand moves continuously over time while the system processes frames.
* **Continuous**: Spatial joint coordinates and confidence probabilities are continuous real numbers.
* **Single Agent**: Only one agent observes the scene and makes decisions.

### 3.3 Agent Architecture
The agent is designed as a **Model-Based Reflex Agent**. A simple reflex agent looks only at the single current frame. This causes severe screen flickering because a camera frame can drop or glitch for a fraction of a second. 

Our agent maintains an internal sliding memory of the last 10 predictions (`MEMORY_SIZE = 10`). It takes a majority vote over this buffer before confirming an action. If the classification confidence drops below 0.6, the agent outputs "unsure" rather than guessing blindly.

---

## 4. Dataset Collection and Preprocessing

### 4.1 Selected Alphabet Signs
A total of 31 distinct static alphabet signs were recorded from the official Family Educational Services Foundation (FESF) Pakistan Sign Language chart:

| Sign Label | Urdu Character | Description |
|---|---|---|
| `alif` | ا | Closed fist with thumb placed alongside the fingers |
| `bay` | ب | Open flat palm with thumb folded across the palm |
| `pay` | پ | Index and middle fingers pointing straight downward |
| `tay` | ت | Sideways facing hand with thumb tucked inward |
| `ttay` | ٹ | Flat sideways hand with thumb pointing straight up |
| `say` | ث | Curved claw shape with fingers bent forward |
| `chay` | چ | Flat sideways horizontal hand |
| `hey` | ح | Index and middle fingers straight up together |
| `khey` | خ | Index and middle fingers crossed, pointing up |
| `daal` | د | Curled fingers with index finger slightly raised |
| `ddal` | ڈ | Index and middle fingers raised with thumb holding other fingers |
| `zaal` | ذ | Curled fist with index finger slightly raised |
| `ray` | ر | Index and middle fingers crossed, pointing straight up |
| `zay` | ز | Index and pinky fingers extended upward ("horns" shape) |
| `zhe` | ژ | Index finger pointing forward with thumb extended |
| `seen` | س | Index and pinky fingers raised, thumb over middle fingers |
| `sheen` | ش | Open hand with spread fingers pointing sideways |
| `suaad` | ص | Three fingers raised with index, middle, and ring fingers |
| `zuaad` | ض | Index, middle, and ring fingers raised, thumb over pinky |
| `toyen` | ط | Index finger pointing straight downward |
| `zoe` | ظ | Index and middle fingers spread in a forward V shape |
| `ain` | ع | Sideways hand with thumb pointing upward |
| `fey` | ف | Closed fist with index finger curled forming a loop with thumb |
| `qaaf` | ق | Closed fist with thumb wrapped across front |
| `kaaf` | ک | Hand curved into a distinct C shape |
| `gaaf` | گ | Curved claw or hook shape facing forward |
| `laam` | ل | Distinct L shape with index finger up and thumb extended |
| `meem` | م | Index finger pointing sideways with thumb pointing down |
| `noon` | ن | Index finger extended pointing straight sideways |
| `wao` | و | Circle formed by touching index tip to thumb tip |
| `choti_yae` | ی | Hand curled with thumb and index in a small loop |
| `yae` | ے | Pinky and thumb extended outward (Y shape) |

### 4.2 Recording Rules and Privacy
Data collection was conducted using the custom interactive tool in `src/collect.py`. The recording process followed strict design constraints:
* **Privacy by Design**: No face images, background photos, or raw video files are stored. Only numerical landmark coordinates are saved.
* **Natural Unflipped Input**: Frames were recorded without horizontal mirroring to preserve true anatomical joint relationships.
* **Environmental Variety**: Samples were recorded across multiple angles, varying distances from the lens (0.5 to 1.5 meters), and different room lighting conditions.
* **Dataset Size**: A total of 625 raw samples were recorded across the 31 alphabet classes.

### 4.3 Data Format
The dataset is stored in `data/raw/psl_data.csv`. Each sample occupies one row with exactly 64 columns:
* Columns 1 through 63: Three coordinates (x, y, z) for each of the 21 hand landmarks (21 * 3 = 63).
* Column 64: The ground-truth text label (e.g. `alif`).

Data is kept completely raw on disk. Normalisation happens entirely in computer memory at load time. This ensures that raw measurements are never lost if the normalisation formula changes.

---

## 5. Feature Engineering and Normalisation

Raw camera coordinates depend heavily on where the user holds their hand on screen. If a hand moves from the top-left corner to the bottom-right corner, every raw coordinate changes completely, even if the hand shape is identical.

To solve this, the `HandSample` class in `src/features.py` applies a two-step normalisation process:

### Step 1: Position Invariance (Translation)
The wrist (Landmark 0) is chosen as the local reference origin. The (x, y, z) coordinates of the wrist are subtracted from each of the 21 joints:

    x'_i = x_i - x_wrist
    y'_i = y_i - y_wrist
    z'_i = z_i - z_wrist

After this step, the wrist is always at (0.0, 0.0, 0.0), regardless of where the hand is located in the video frame.

### Step 2: Scale Invariance (Size and Distance)
To make the representation independent of hand size and distance from the camera, the system calculates the 3D Euclidean distance from the wrist to every joint:

    d_i = √((x'_i)² + (y'_i)² + (z'_i)²)

The maximum distance across all 21 points (d_max) represents the total physical span of the hand. Every coordinate is divided by d_max:

    x''_i = x'_i / d_max
    y''_i = y'_i / d_max
    z''_i = z'_i / d_max

All coordinates are now strictly scaled between -1.0 and +1.0.

### Step 3: Feature Vector Construction
Because Landmark 0 (the wrist) is now guaranteed to be (0.0, 0.0, 0.0) for every sample, it provides no discriminative information to machine learning algorithms. The wrist is dropped, leaving the remaining 20 joints. 

Multiplying 20 joints by 3 coordinates yields a flat vector of **60 informative numbers**. The `tovector()` method returns this 60-element array for model training.

---

## 6. Machine Learning Models

The project implements an object-oriented hierarchy in `src/models.py`. The abstract parent class `BaseRecognizer` defines standard methods: `build()`, `fit()`, `predict()`, and `score()`. Four models inherit from this base:

### 6.1 Baseline Recognizer (Dummy Classifier)
* **How it works**: Always predicts the most frequent class in the training set regardless of the input.
* **Purpose**: Serves as a scientific sanity check. In a balanced 31-class dataset, random guessing or majority guessing achieves roughly 3.2% to 4.0% accuracy. Any legitimate machine learning model must perform significantly higher than this baseline.

### 6.2 K-Nearest Neighbors (KNN)
* **Configuration**: k = 5 neighbors, with distance-based weighting.
* **How it works**: Stores all training samples. When a new hand is observed, it measures the Euclidean distance to all stored hands and selects the 5 closest neighbors.
* **Why distance weighting**: Points that are very close to the test sample receive a higher voting weight than points further away. This prevents borderline samples from distorting the decision.

### 6.3 Support Vector Machine (SVM)
* **Configuration**: Radial Basis Function (RBF) kernel, regularization parameter C = 10, probability estimation enabled.
* **How it works**: Maps the 60-dimensional features into a higher-dimensional space where geometric decision boundaries (hyperplanes) separate all 31 alphabet classes with maximum margin.
* **Why RBF kernel**: Hand sign boundaries in joint space are non-linear. The RBF kernel allows flexible, curved boundaries that generalize well to natural hand posture variations.

### 6.4 Random Forest Classifier
* **Configuration**: 200 decision trees (`n_estimators = 200`).
* **How it works**: Trains an ensemble of 200 decision trees, each seeing a random subset of training samples and features. The final prediction is determined by majority vote across all trees.
* **Why an ensemble**: Individual decision trees are prone to overfitting on continuous coordinate data. Combining 200 trees reduces variance and yields robust predictions.

---

## 7. Experimental Setup and Results

### 7.1 Cross-Validation Strategy
To prevent cherry-picked results, models were evaluated across 20 distinct seeded train-test splits (`range(20)`). 

For each split:
* 80% of the data is used for training and 20% is held out for testing.
* Splitting is stratified (`stratify = y`) so every letter is equally represented in both sets.
* Training time, per-sample prediction time, Classification Accuracy, and Macro F1 score are recorded.
* Sample statistics are calculated using unbiased sample standard deviation (N - 1 degrees of freedom).

### 7.2 Results Table
The table below summarizes the performance of all four models over the 20 seeded splits across the 31 alphabet classes:

| Model | Accuracy (Mean ± Std %) | Macro F1 (Mean ± Std %) | Training Time (ms) | Prediction Time (μs/sample) |
|---|---|---|---|---|
| **Baseline (Dummy)** | 4.00 ± 0.00% | 0.25 ± 0.00% | 1.13 ms | 1.39 μs |
| **KNN** | 95.64 ± 1.55% | 95.54 ± 1.61% | 1.82 ms | 318.93 μs |
| **Support Vector Machine (SVM)** | **97.72 ± 1.25%** | **97.66 ± 1.31%** | 213.19 ms | **132.68 μs** |
| **Random Forest** | 95.84 ± 2.05% | 95.70 ± 2.15% | 1778.89 ms | 311.81 μs |

### 7.3 Analysis of Results
1. **High Class Separability**: All three machine learning models achieved over 95.5% average accuracy, outperforming the 4.00% baseline by more than 91 percentage points. SVM led all models with **97.72% accuracy** and **97.66% Macro F1**, demonstrating that even across 31 alphabet classes, our 60 normalised geometric features clearly distinguish hand shapes.
2. **Speed Comparison**: SVM achieved a very low prediction latency of **132.68 microseconds per sample**. This is more than twice as fast as KNN (318 μs) and Random Forest (311 μs). In a 30 to 60 FPS live video application, an inference budget of 132 microseconds leaves ample time for camera capture and UI rendering.
3. **Stability**: The standard deviation across all 20 seeds remained below 1.3% for SVM. This tight distribution confirms that the system delivers consistent performance regardless of how the dataset is split.

### 7.4 Generated Visual Charts
Three figures were automatically generated and saved to `outputs/figures/`:
* **Figure 1 (Accuracy Comparison with Error Bars)**: Visually illustrates the performance gap between the 4% baseline and the ~97.7% AI models, with error bars confirming tight stability.
* **Figure 2 (Confusion Matrix)**: A 31 by 31 heatmap showing near-perfect identification across all 31 classes, confirming minimal confusion between similar hand shapes.
* **Figure 3 (Learning Curve)**: Demonstrates that training and validation accuracy converge smoothly as sample size increases, confirming that the models generalize well.

---

## 8. Discussion of Limitations and Data Leakage

### 8.1 Temporal Autocorrelation (Data Leakage)
A critical insight of this study relates to how video data behaves. When a webcam records 20 frames of a hand sign within a 2-second window, consecutive frames are nearly identical. 

In a standard random train-test split, adjacent frames from the same recording session end up in both the training set and the test set. The model is essentially tested on near-duplicate images of what it just saw. 

While the 97.7% accuracy demonstrates that the mathematical pipeline is correct, this number is optimistic. Testing the model on a completely new session recorded on a different day, in a different room, or with a different person will naturally experience a modest drop in performance. Acknowledging this distinction is essential for honest machine learning reporting.

### 8.2 Physical and Hardware Constraints
* **Low Light Sensitivity**: MediaPipe relies on visual contrast to identify hand joints. In dimly lit environments, landmark tracking degrades.
* **Single Signer Scope**: As a solo project, data was collected from one primary signer. Cross-subject validation across signers with varying hand shapes represents the next research step.

---

## 9. Future Work

1. **Vocabulary Completion**: Adding the remaining dynamic motion signs using temporal sequence modeling (such as LSTMs or Temporal Convolutional Networks).
2. **Two-Handed Signs**: Expanding the tracking pipeline to support two-handed PSL vocabulary.
3. **Cross-Signer Testing**: Recording benchmark sets from multiple signers of different ages and genders to improve universal generalization.
4. **Mobile Deployment**: Packaging the trained SVM model into an Android or iOS application for on-device real-time communication.

---

## 10. Individual Contribution

* **Areesha Ahmad (24K-3043)**: 100% individual contribution. Designed the project architecture, collected the complete 31-sign dataset, wrote the normalisation mathematics, built the machine learning classes, developed the smoothing agent and word builder, authored automated unit tests, designed the CustomTkinter GUI, and prepared all project documentation.

---

## 11. Conclusion

This project successfully demonstrates that a lightweight machine learning pipeline combining MediaPipe Hands, coordinate normalisation, and a Support Vector Machine can accurately translate Pakistan Sign Language alphabet signs in real time. Across 31 distinct alphabet classes and 625 samples, the SVM achieved 97.72% mean test accuracy with an inference latency of just 132 microseconds per sample. Coupled with a model-based reflex smoothing agent and an interactive CustomTkinter interface, the system provides a robust, accessible foundation for digital PSL assistance in Pakistan.
