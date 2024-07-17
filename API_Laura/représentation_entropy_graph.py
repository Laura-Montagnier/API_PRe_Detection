#!/usr/bin/env python

import os
import argparse
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt

def calculate_byte_entropies(filename):
    try:
        with open(filename, 'rb') as file:
            byte_data = file.read()
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return []

    byte_frequencies = np.zeros(256, dtype=int)

    for byte in byte_data:
        byte_frequencies[byte] += 1

    total_bytes = len(byte_data)
    if total_bytes == 0:
        return []

    byte_probabilities = byte_frequencies / total_bytes

    entropies = [entropy([prob, 1 - prob], base=2) if prob > 0 else 0 for prob in byte_probabilities]

    return entropies

def plot_entropies(entropies, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(entropies)), entropies, color='black', linestyle='-')
    plt.xlabel('Byte Value (0-255)')
    plt.ylabel('Entropy')
    plt.title('Entropy of Each Byte Value')
    plt.savefig(output_path)
    plt.close()

def main(folder_path):
    output_dir = os.path.expanduser('~/API_Laura/Résultats/Graphes_entropie')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(folder_path):
        print(f"Le dossier {folder_path} n'existe pas.")
        return

    for binary in os.listdir(folder_path):
        file_path = os.path.join(folder_path, binary)
        if os.path.isfile(file_path):  # Ensure it's a file
            entropies = calculate_byte_entropies(file_path)
            if entropies:
                output_path = os.path.join(output_dir, f"{os.path.splitext(binary)[0]}_entropy.png")
                plot_entropies(entropies, output_path)
                print(f"Graphe d'entropie enregistré pour {file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate and plot byte entropy of binary files in a folder.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing binary files.')
    
    args = parser.parse_args()
    main(args.folder_path)





