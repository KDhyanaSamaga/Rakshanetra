# <span style="color:#2ecc71">RAKSHANETRA</span> 🛡️👁️

## <span style="color:#27ae60">Real-Time Animal Intrusion Detection & Alert Platform</span>

---

<div align="center">

**Department of Artificial Intelligence & Machine Learning**

</div>

---

## 🌍 Overview

**RakshaNetra** is an **AI-powered, real-time wildlife intrusion detection and alert system** designed to mitigate **human–wildlife conflict** in forest-adjacent villages. The platform continuously monitors village boundaries using **stationary cameras** and applies **computer vision (YOLOv5)** to detect and classify wild animals such as **Elephants, Leopards, and Wild Boars**.

Upon detecting an intrusion, the system instantly sends **Telegram alerts** with images, timestamps, and location context to forest officials, enabling **proactive intervention** and improved safety for both humans and wildlife.

---

## 🎯 Problem Statement

Forest-bordering villages frequently experience dangerous wildlife intrusions that lead to:

* Crop destruction
* Property damage
* Injuries and loss of human life

Traditional methods (electric fences, flash cameras, trenches) are either unsafe, ineffective, or lack **real-time alerting**. RakshaNetra addresses this gap with a **non-invasive, intelligent, and scalable solution**.

---

## ✅ Objectives

* Develop a **real-time animal detection system** using AI and computer vision
* Accurately **classify wild animals** using YOLOv5
* Implement **instant alert mechanisms** via Telegram
* Log intrusion events with **time, image, and animal type**
* Reduce human–wildlife conflict and enhance rural safety

---

## 🧠 Key Features

* 🎥 **Live Video Monitoring** (Day & Night)
* 🐘 **Multi-Animal Detection** (Elephant, Leopard, Wild Boar)
* 🚧 **Virtual Boundary Mapping**
* ⚡ **Instant Telegram Alerts with Images**
* 🧾 **Event Logging for Analysis**
* 🖥️ **Edge AI Deployment (Raspberry Pi)**

---

## 🏗️ System Architecture

RakshaNetra follows an **edge-computing architecture**:

* **Camera Module**: Continuous video capture
* **Edge Device (Raspberry Pi)**:

  * Frame capture & preprocessing (OpenCV)
  * YOLOv5 inference engine
  * Intrusion decision logic
  * Alert manager (Telegram Bot API)
* **Notification Layer**: Real-time alerts to authorities

---

## ⚙️ Technology Stack

### 🔧 Hardware

* Night Vision Surveillance Camera
* Raspberry Pi (Edge Device)

### 💻 Software

* **Language**: Python 3.x
* **Frameworks & Libraries**:

  * YOLOv5 (Object Detection)
  * OpenCV (Image Processing)
  * NumPy, Matplotlib
* **Alert System**: Telegram Bot API

---

## 🧪 Methodology

1. **Data Collection**

   * Real-world footage (Pilikula Biological Park)
   * Online wildlife videos

2. **Data Annotation & Preprocessing**

   * Manual bounding box annotation using Roboflow
   * Augmentation: flip, rotate, brightness, noise

3. **Model Training**

   * YOLOv5 with custom dataset (~6000 images)
   * Hyperparameter tuning for accuracy

4. **Evaluation Metrics**

   * Precision: 82%
   * Recall: 81%
   * mAP@0.5: 85%

---

## 🚨 Intrusion Detection Logic

* A **virtual boundary line** represents the village border
* For each detected animal:

  * Compute bottom-center of bounding box
  * Check boundary crossing

**Binary Decision Model:**

* ❌ Not Crossed → No Alert
* ✅ Crossed → Immediate Alert

---

## 📲 Alert System

* Telegram alerts include:

  * Detected animal type
  * Timestamp
  * Camera/location ID
  * Compressed image

Alerts continue as long as the animal remains inside the boundary.

---

## 🧪 Testing

### Black Box Testing

* Validated detection accuracy
* Verified boundary logic
* Confirmed real-time alerting

### White Box Testing

* Frame processing validation
* Boundary-crossing logic testing
* Alert module non-blocking checks

**Result:** All tests passed successfully.

---

## 📊 Results

* High accuracy even in low-light conditions
* Real-time performance with minimal latency
* Stable deployment on Raspberry Pi
* Effective alerts enabling quick response

---

## 🚀 Future Scope

* ☀️ Solar-powered IoT camera units
* 🚁 Drone-based wildlife tracking
* ⚡ Edge AI accelerators (Jetson Nano, Coral TPU)
* 📍 GPS-based animal movement tracking
* 📱 Mobile application integration

---

## 👥 Project Team

* **Bhargavi Kulkarni** – 4SN22AI018
* **Janma A Saragodu** – 4SN22AI029
* **K Dhyana Samaga** – 4SN22AI031
* **Vanashree** – 4SN22AI061

---

## 🧭 Ethics & Permissions

All data used in this project was obtained **ethically** from publicly available sources or collected with **prior authorization** from wildlife officials. The system strictly adheres to wildlife conservation and data privacy guidelines.

---

## 📜 License

This project is intended for **academic and research purposes**. Commercial use requires prior permission.

---

<div align="center">

### 🌱 *Protecting Communities. Preserving Wildlife.*

</div>
