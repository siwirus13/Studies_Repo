#!/usr/bin/env python3

import os
import subprocess


mapper_wordcount = "mapper_wordcount.py"
reducer_wordcount = "reducer_wordcount.py"
mapper_chars = "mapper_chars.py"
mapper_chars2 = "mapper_chars_opt.py"
reducer_chars = "reducer_chars.py"

input_file = "localfile.txt"

def ensure_hdfs_input():
    print("➤ Tworzenie katalogu input w HDFS i przesyłanie pliku...")
    subprocess.run(["hdfs", "dfs", "-mkdir", "-p", "input"], stderr=subprocess.DEVNULL)
    subprocess.run(["hdfs", "dfs", "-put", "-f", input_file, "input"])


import subprocess

def run_mapreduce(mapper, reducer, output_dir):
    print(f"\n➤ Uruchamianie MapReduce: {mapper} + {reducer} → {output_dir}")
    subprocess.run(["hdfs", "dfs", "-rm", "-r", "-f", output_dir], stderr=subprocess.DEVNULL)

    hadoop_home = os.environ.get("HADOOP_HOME")
   
    streaming_dir = os.path.join(hadoop_home, "share/hadoop/tools/lib")
    jars = [f for f in os.listdir(streaming_dir) if f.startswith("hadoop-streaming") and f.endswith(".jar")]
    if not jars:
        print(f"❌ Nie znaleziono hadoop-streaming-*.jar w {streaming_dir}")
        return
    streaming_jar = os.path.join(streaming_dir, jars[0])

    
    res = subprocess.run([
        "hadoop", "jar", streaming_jar,
        "-files", f"{mapper},{reducer}",
        "-mapper", f"python3 {mapper}",
        "-reducer", f"python3 {reducer}",
        "-input", "input",
        "-output", output_dir
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    print("-----> STDOUT:")
    print(res.stdout)
    print("-----> STDERR:")
    print(res.stderr)
    print(res.returncode, "← kod return")

    if res.returncode != 0:
        print(f"❌ MapReduce job dla {output_dir} zakończony błędem, return code {res.returncode}")
        return False

    print(f"➤ Wynik dla {output_dir}:")
    cat = subprocess.run(["hdfs", "dfs", "-cat", f"{output_dir}/part-00000"], capture_output=True, text=True)
    print(cat.stdout)
    return True


def zadanie1():
    ensure_hdfs_input()

def zadanie2():
    run_mapreduce(mapper_wordcount, reducer_wordcount, "output_words")

def zadanie3():
    run_mapreduce(mapper_chars, reducer_chars, "output_chars")

def zadanie4():
    run_mapreduce(mapper_chars2, reducer_chars, "output_chars")


if __name__ == "__main__":
    #zadanie1()
    #zadanie2()
    zadanie3()
    zadanie4()
