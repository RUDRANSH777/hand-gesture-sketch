#  Hand Gesture Sketch

A simple and interactive **computer vision project** that lets you draw in the air using just your fingers.

Using your webcam, the application tracks your hand in real time and converts **pinch gestures into smooth drawings** on a virtual canvas — creating a completely touchless whiteboard experience.

Built with **Python, OpenCV, and MediaPipe**.

---

## 🚀 Features

- Real-time hand tracking
- Pinch gesture to draw
- Smooth and stable line rendering
- Clear canvas button
- Lightweight and responsive
- Works with any standard webcam
- Exit using ESC or Q

---

## 🧠 How It Works

The application follows a simple pipeline:

1. Captures live video from the webcam  
2. Detects 21 hand landmarks using MediaPipe  
3. Measures the distance between thumb and index finger  
4. Pinch gesture activates drawing  
5. Lines are drawn on a transparent canvas overlay  
6. Pinch on CLEAR resets the canvas  

This creates a natural and intuitive gesture-based drawing system powered by computer vision.

---

## 🛠 Tech Stack

- Python  
- OpenCV  
- MediaPipe Hands  
- NumPy  

---

## 📦 Installation

### Clone the repository
```bash
git clone https://github.com/RUDRANSH777/hand-gesture-sketch.git
cd hand-gesture-sketch
