import subprocess
import sys
import os

def convert_glsl_to_spirv(input_file, output_file):
    command = f"glslangValidator -V {input_file} -o {output_file}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error converting GLSL to SPIR-V:", result.stderr.decode('utf-8'))
    else:
        print("Conversion successful. Output saved to", output_file)

def convert_opencl_to_spirv(input_file, output_file):
    command = f"clspv {input_file} -o {output_file}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error converting OpenCL to SPIR-V:", result.stderr.decode('utf-8'))
    else:
        print("Conversion successful. Output saved to", output_file)

def convert_shader_to_spirv(input_file):
    extension = os.path.splitext(input_file)[1]
    output_file = os.path.splitext(input_file)[0] + ".spv"

    if extension in [".glsl", ".vert", ".frag", ".geom", ".comp"]:
        convert_glsl_to_spirv(input_file, output_file)
    elif extension == ".cl":
        convert_opencl_to_spirv(input_file, output_file)
    else:
        print("Unsupported file type. Please provide a .glsl, .vert, .frag, .geom, .comp, or .cl file.")

if len(sys.argv) != 2:
    print("Usage: python convert_to_spirv.py <input_file>")
else:
    input_file = sys.argv[1]
    convert_shader_to_spirv(input_file)
