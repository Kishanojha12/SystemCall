# AI-Powered System Call Optimization Dashboard

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green?logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-AI%20Model-orange?logo=tensorflow)



---

## 🚀 Overview

This project is an **AI-powered system call latency optimizer** designed to monitor system calls, analyze their latency, and predict improved latency using a machine learning model (LSTM). It includes a **Flask-based dashboard** for interactive visualization.

---

## 📂 Project Structure

```bash
├── abc.py                  # Collects syscall logs into syscall_logs.csv
├── optimize.py            # AI model to optimize syscall latency
├── dashboard.py           # Flask-based interactive dashboard
├── templates/
│   └── dashboard.html     # Dashboard UI template
├── syscall_logs.csv       # Input: Raw syscall data (generated)
├── optimized_logs.csv     # Output: Optimized syscall data (generated)
├── requirements.txt       # Required Python libraries
```

---

## 💡 Features

- ✅ **System Call Monitoring** via strace (manually or through script)
- ✅ **LSTM-based Latency Prediction** using TensorFlow
- ✅ **Clean Visualization** with Flask and Matplotlib
- ✅ **Interactive Dashboard** with filtering (PID, Time)
- ✅ **Before vs After Comparison** (Charts + Table)

---

## 🧠 How It Works

### Step 1: Collect System Call Logs

Use `abc.py` to generate syscall logs (or create your own CSV in the format):

```csv
syscall,frequency,latency,pid,time
read,23,0.004,1234,2024-04-01 10:10:00
write,10,0.007,1234,2024-04-01 10:10:01
```

### Step 2: Run the Optimizer

Train the LSTM model and generate `optimized_logs.csv`:

```bash
python optimize.py
```

### Step 3: Start the Dashboard

Launch the Flask dashboard:

```bash
python dashboard.py
```

Then visit: `http://127.0.0.1:5000`

---

## 📊 Dashboard UI

- **Line Chart**: Latency Before vs After over time
- **Bar Chart**: Average Before vs After latency
- **Table**: Filterable syscall log entries
- **Filters**: PID and time range

---

## 🛠️ Requirements

```txt
flask
pandas
matplotlib
tensorflow
scikit-learn
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 📌 Use Cases

- OS performance analysis and tuning
- Real-time syscall latency monitoring
- ML-based syscall behavior research
- Educational visualization for operating systems

---

## 🧑‍💻 Author

**Kishan Ojha**  
**Gaurav Kumar**  
**Jatin Kumar Prajapati**  

---

## 📎 License

This project is open-source under the MIT License.

---

## 🙌 Support or Contribute

Found a bug? Have suggestions? Feel free to open an issue or PR. Contributions are welcome!
