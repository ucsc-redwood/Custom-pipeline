import subprocess

# Function to execute a shell command and return its output
def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    output, _ = process.communicate()
    return output.decode("utf-8").strip()

# Print CPU name
cpu_name = run_command("grep 'model name' /proc/cpuinfo | uniq | cut -d ':' -f 2")
print("CPU:", cpu_name)

# Print GPU name
gpu_name = run_command("lspci | grep -i 'VGA\|3D\|Display'")
print("GPU:", gpu_name)

