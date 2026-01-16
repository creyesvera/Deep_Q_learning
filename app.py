import subprocess
import webbrowser
import time

# lanzar servidor
subprocess.Popen(["python", "-m", "http.server", "8000"])

# esperar un poco
time.sleep(1)

# abrir navegador
webbrowser.open("http://localhost:8000/dashboard.html")
