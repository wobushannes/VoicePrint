#!/usr/bin/env python3
"""
VoicePrint - FORENSIC ATTRIBUTION (NO LIBROSA - FINAL)
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import torch
import numpy as np
from scipy.spatial.distance import cosine
from scipy import signal
from scipy.fftpack import dct
from speechbrain.inference.speaker import EncoderClassifier
import soundfile as sf
import scipy.signal
from pathlib import Path
import json
from datetime import datetime

# ============================================================================
# CUDA CHECK
# ============================================================================
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Using device: {device}")

# ============================================================================
# AUDIO LOADING
# ============================================================================
def load_audio(path):
    data, sr = sf.read(path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    if sr != 16000:
        new_len = int(len(data) * 16000 / sr)
        data = scipy.signal.resample(data, new_len)
        sr = 16000
    return torch.from_numpy(data).float().unsqueeze(0), sr

# ============================================================================
# ATTRIBUTE ANALYSIS (100% NO LIBROSA)
# ============================================================================
def analyze_attributes(path):
    """Extract voice attributes using only numpy/scipy"""
    try:
        data, sr = sf.read(path)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        if sr != 16000:
            new_len = int(len(data) * 16000 / sr)
            data = scipy.signal.resample(data, new_len)
            sr = 16000
        
        # 1. MFCCs
        mfccs = compute_mfcc(data, sr)
        mfcc_mean = np.mean(mfccs, axis=0)
        
        # 2. Pitch
        pitch = compute_pitch(data, sr)
        
        # 3. Energy
        energy = np.sqrt(np.mean(data**2))
        
        # 4. Spectral centroid
        f, t, Zxx = signal.stft(data, fs=sr, nperseg=1024)
        magnitude = np.abs(Zxx)
        frequencies = np.linspace(0, sr/2, magnitude.shape[0])
        spectral_centroid = np.sum(frequencies[:, None] * magnitude, axis=0) / np.sum(magnitude, axis=0)
        spectral_centroid = np.nanmean(spectral_centroid)
        
        return {
            "pitch": float(pitch),
            "mfcc": mfcc_mean.tolist(),
            "energy": float(energy),
            "spectral_centroid": float(spectral_centroid)
        }
    except Exception as e:
        print(f"Error in attribute analysis: {e}")
        return None

def compute_mfcc(data, sr, num_cep=13):
    """Simple MFCC computation - NO LIBROSA"""
    pre_emphasis = 0.97
    data = np.append(data[0], data[1:] - pre_emphasis * data[:-1])
    
    frame_size = int(0.025 * sr)
    hop_size = int(0.010 * sr)
    window = np.hamming(frame_size)
    
    num_frames = (len(data) - frame_size) // hop_size + 1
    frames = np.zeros((num_frames, frame_size))
    for i in range(num_frames):
        start = i * hop_size
        frames[i] = data[start:start+frame_size] * window
    
    fft = np.fft.rfft(frames, n=512)
    magnitude = np.abs(fft)
    
    num_mel = 26
    mel_filters = np.zeros((num_mel, magnitude.shape[1]))
    mel_points = np.linspace(0, 2595 * np.log10(1 + sr/2/700), num_mel + 2)
    hz_points = 700 * (10**(mel_points/2595) - 1)
    bins = np.floor((magnitude.shape[1] - 1) * hz_points / (sr/2)).astype(int)
    
    for i in range(1, num_mel + 1):
        mel_filters[i-1, bins[i-1]:bins[i]] = np.linspace(0, 1, bins[i] - bins[i-1])
        mel_filters[i-1, bins[i]:bins[i+1]] = np.linspace(1, 0, bins[i+1] - bins[i])
    
    mel_energy = np.dot(magnitude, mel_filters.T)
    mel_energy = np.maximum(mel_energy, 1e-10)
    
    log_energy = np.log(mel_energy)
    mfcc = np.apply_along_axis(lambda x: dct(x, type=2, norm='ortho'), 1, log_energy)
    mfcc = mfcc[:, :num_cep]
    
    return mfcc

def compute_pitch(data, sr):
    """Pitch estimation using autocorrelation"""
    corr = np.correlate(data, data, mode='full')
    corr = corr[len(corr)//2:]
    
    min_period = int(sr / 600)
    max_period = int(sr / 75)
    
    if len(corr) <= max_period:
        return 0
    
    corr_slice = corr[min_period:max_period]
    if len(corr_slice) == 0:
        return 0
    
    peak_idx = np.argmax(corr_slice) + min_period
    pitch = sr / peak_idx if peak_idx > 0 else 0
    return pitch

def compare_attributes(attr_clone, attr_ref):
    """Compare attributes and return similarity scores"""
    if attr_clone is None or attr_ref is None:
        return None
    
    pitch_diff = abs(attr_clone["pitch"] - attr_ref["pitch"])
    pitch_sim = max(0, 1 - (pitch_diff / 200))
    
    mfcc_sim = 1 - cosine(attr_clone["mfcc"], attr_ref["mfcc"])
    
    energy_diff = abs(attr_clone["energy"] - attr_ref["energy"])
    energy_sim = max(0, 1 - (energy_diff / (attr_clone["energy"] + attr_ref["energy"] + 1e-6)))
    
    centroid_diff = abs(attr_clone["spectral_centroid"] - attr_ref["spectral_centroid"])
    centroid_sim = max(0, 1 - (centroid_diff / 5000))
    
    return {
        "pitch": pitch_sim * 100,
        "mfcc": mfcc_sim * 100,
        "energy": energy_sim * 100,
        "centroid": centroid_sim * 100
    }

# ============================================================================
# GLOBALS
# ============================================================================
clone_file = None
folder_a = None
folder_b = None
model_ecapa = None
export_data = None
analyze_attributes_enabled = True

# ============================================================================
# LOAD MODEL
# ============================================================================
def load_model():
    global model_ecapa
    status_label.config(text="Loading ECAPA-TDNN...")
    root.update()
    try:
        model_ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
        status_label.config(text="Model loaded!")
        load_button.config(state="disabled", text="✓")
        check_ready()
        messagebox.showinfo("Success", "Model loaded successfully!")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# ============================================================================
# FILE/FOLDER SELECTION
# ============================================================================
def select_clone():
    global clone_file
    f = filedialog.askopenfilename(filetypes=[("Audio", "*.mp3 *.wav *.flac")])
    if f:
        clone_file = f
        clone_label.config(text=f"✓ {os.path.basename(f)}")
        check_ready()

def select_folder_a():
    global folder_a
    d = filedialog.askdirectory()
    if d:
        folder_a = d
        count = len(list(Path(d).glob("*.mp3")) + list(Path(d).glob("*.wav")))
        folder_a_label.config(text=f"✓ {os.path.basename(d)} ({count} files)")
        check_ready()

def select_folder_b():
    global folder_b
    d = filedialog.askdirectory()
    if d:
        folder_b = d
        count = len(list(Path(d).glob("*.mp3")) + list(Path(d).glob("*.wav")))
        folder_b_label.config(text=f"✓ {os.path.basename(d)} ({count} files)")
        check_ready()

def check_ready():
    if clone_file and folder_a and folder_b and model_ecapa is not None:
        compare_button.config(state="normal")
        export_button.config(state="normal")

# ============================================================================
# EXTRACT EMBEDDING
# ============================================================================
def extract_ecapa(path):
    signal, sr = load_audio(path)
    signal = signal.to(device)
    emb = model_ecapa.encode_batch(signal)
    return emb.squeeze().detach().cpu().numpy()

# ============================================================================
# TOGGLE ATTRIBUTE ANALYSIS
# ============================================================================
def toggle_attributes():
    global analyze_attributes_enabled
    analyze_attributes_enabled = not analyze_attributes_enabled
    status_label.config(text=f"Attribute analysis: {'ON' if analyze_attributes_enabled else 'OFF'}")

# ============================================================================
# COMPARE
# ============================================================================
def compare_with_explanation():
    def run():
        try:
            compare_button.config(state="disabled", text="Processing...")
            status_label.config(text="Extracting clone embeddings...")
            root.update()
            
            emb_clone = extract_ecapa(clone_file)
            
            if analyze_attributes_enabled:
                attr_clone = analyze_attributes(clone_file)
            else:
                attr_clone = None
            
            files_a = list(Path(folder_a).glob("*.mp3")) + list(Path(folder_a).glob("*.wav"))
            files_b = list(Path(folder_b).glob("*.mp3")) + list(Path(folder_b).glob("*.wav"))
            
            results = []
            explanations = {}
            
            all_files = files_a + files_b
            for i, f in enumerate(all_files):
                status_label.config(text=f"Processing {i+1}/{len(all_files)}: {f.name}")
                root.update()
                
                emb_ref = extract_ecapa(str(f))
                
                # FIX: cosine similarity properly clamped
                sim_raw = 1 - cosine(emb_clone, emb_ref)
                sim = max(0, min(100, sim_raw * 100))
                
                if analyze_attributes_enabled:
                    attr_ref = analyze_attributes(str(f))
                    attr_sim = compare_attributes(attr_clone, attr_ref)
                else:
                    attr_sim = None
                
                folder = "A" if f in files_a else "B"
                results.append((f.name, sim, folder))
                explanations[f.name] = attr_sim
            
            results.sort(key=lambda x: x[1], reverse=True)
            
            scores_b = [s for n, s, f in results if f == "B"]
            median_b = np.median(scores_b) if scores_b else 0
            std_b = np.std(scores_b) if scores_b else 0
            threshold = median_b + 2 * std_b
            
            classified = []
            for name, score, folder in results:
                if folder == "A":
                    if score > threshold:
                        status = "🟢 ENTHALTEN"
                    else:
                        status = "🟡 UNSICHER"
                else:
                    if score < threshold:
                        status = "🔴 NICHT ENTHALTEN"
                    else:
                        status = "🟡 UNSICHER (ähnlich)"
                classified.append((name, score, folder, status))
            
            text = "📊 VOICEPRINT - FORENSIC ATTRIBUTION\n"
            text += "=" * 90 + "\n"
            text += f"📌 STATISTICS:\n"
            text += f"   Median (Folder B): {median_b:.1f}%\n"
            text += f"   StdDev (Folder B): {std_b:.1f}%\n"
            text += f"   Threshold: {threshold:.1f}%\n"
            text += f"   Attribute Analysis: {'ON' if analyze_attributes_enabled else 'OFF'}\n"
            text += "=" * 90 + "\n\n"
            
            text += "  %     | Folder | Status                    | File\n"
            text += "-" * 90 + "\n"
            
            for name, score, folder, status in classified:
                text += f"{score:5.1f}% |   {folder}   | {status:25} | {name}\n"
            
            if analyze_attributes_enabled:
                text += "\n" + "=" * 90 + "\n"
                text += "🔍 EXPLANATIONS FOR UNCERTAIN FILES:\n"
                text += "-" * 90 + "\n"
                
                for name, score, folder, status in classified:
                    if "UNSICHER" in status:
                        attr_sim = explanations.get(name, {})
                        if attr_sim:
                            text += f"\n📁 {name} ({score:.1f}%):\n"
                            text += f"   🎵 Pitch (Tonhöhe):      {attr_sim.get('pitch', 0):.1f}% match\n"
                            text += f"   🎹 MFCC (Klangfarbe):    {attr_sim.get('mfcc', 0):.1f}% match\n"
                            text += f"   ⚡ Energy (Lautstärke):  {attr_sim.get('energy', 0):.1f}% match\n"
                            text += f"   🎵 Spectral Centroid:    {attr_sim.get('centroid', 0):.1f}% match\n"
                            
                            avg_attr = np.mean([attr_sim.get('pitch', 0), attr_sim.get('mfcc', 0), 
                                               attr_sim.get('energy', 0), attr_sim.get('centroid', 0)])
                            if avg_attr > 70:
                                text += f"   💡 Likely similar voice characteristics\n"
                            elif avg_attr > 50:
                                text += f"   💡 Some shared characteristics, but not identical\n"
                            else:
                                text += f"   💡 Different voice characteristics\n"
            else:
                text += "\n" + "=" * 90 + "\n"
                text += "⚠️  Attribute analysis is OFF. Only similarity scores are shown.\n"
            
            global export_data
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "clone": clone_file,
                "folder_a": folder_a,
                "folder_b": folder_b,
                "threshold": threshold,
                "results": classified,
                "explanations": explanations,
                "attribute_analysis_enabled": analyze_attributes_enabled
            }
            
            root.after(0, lambda: result_text.config(state="normal"))
            root.after(0, lambda: result_text.delete("1.0", tk.END))
            root.after(0, lambda: result_text.insert("1.0", text))
            root.after(0, lambda: result_text.config(state="disabled"))
            root.after(0, lambda: status_label.config(text="Done"))
            root.after(0, lambda: compare_button.config(state="normal", text="Compare All"))
            root.after(0, lambda: export_button.config(state="normal"))
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            root.after(0, lambda: messagebox.showerror("Error", str(e)))
            root.after(0, lambda: compare_button.config(state="normal", text="Compare All"))
    
    threading.Thread(target=run, daemon=True).start()

# ============================================================================
# EXPORT
# ============================================================================
def export_results():
    try:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json")]
        )
        if not file_path:
            return
        
        data = export_data
        if not data:
            messagebox.showerror("Error", "No data to export.")
            return
        
        if file_path.endswith('.json'):
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("File,Score,Folder,Status,Pitch,MFCC,Energy,Centroid\n")
                for name, score, folder, status in data["results"]:
                    attr = data["explanations"].get(name, {})
                    f.write(f"{name},{score:.1f},{folder},{status},{attr.get('pitch', 0):.1f},{attr.get('mfcc', 0):.1f},{attr.get('energy', 0):.1f},{attr.get('centroid', 0):.1f}\n")
        
        messagebox.showinfo("Success", f"Exported to: {file_path}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# ============================================================================
# GUI
# ============================================================================
root = tk.Tk()
root.title("VoicePrint - Forensic Attribution")
root.geometry("1000x850")
root.resizable(True, True)

tk.Label(root, text="🎙️ VoicePrint - Forensic Attribution", font=("Arial", 16, "bold"), pady=10).pack()

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

control_frame = tk.Frame(main_frame, width=300)
control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

clone_label = tk.Label(control_frame, text="No clone selected", relief=tk.GROOVE, anchor=tk.W, padx=10, pady=5)
clone_label.pack(fill=tk.X, pady=5)
tk.Button(control_frame, text="Select Clone Audio", command=select_clone).pack(pady=5)

folder_a_label = tk.Label(control_frame, text="No folder A selected", relief=tk.GROOVE, anchor=tk.W, padx=10, pady=5)
folder_a_label.pack(fill=tk.X, pady=5)
tk.Button(control_frame, text="Select Folder A (sources)", command=select_folder_a).pack(pady=5)

folder_b_label = tk.Label(control_frame, text="No folder B selected", relief=tk.GROOVE, anchor=tk.W, padx=10, pady=5)
folder_b_label.pack(fill=tk.X, pady=5)
tk.Button(control_frame, text="Select Folder B (others)", command=select_folder_b).pack(pady=5)

ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)

load_button = tk.Button(control_frame, text="Load Model", command=load_model, bg="#e0e0e0")
load_button.pack(pady=5, fill=tk.X)

compare_button = tk.Button(control_frame, text="Analyze", command=compare_with_explanation, bg="#007acc", fg="white", state="disabled")
compare_button.pack(pady=5, fill=tk.X)

export_button = tk.Button(control_frame, text="Export Results", command=export_results, bg="#cc8800", fg="white", state="disabled")
export_button.pack(pady=5, fill=tk.X)

# Checkbox
attr_var = tk.BooleanVar(value=True)
attr_check = tk.Checkbutton(
    control_frame,
    text="Enable Attribute Analysis (Explanations)",
    variable=attr_var,
    command=toggle_attributes
)
attr_check.pack(pady=5, anchor=tk.W)

info_label = tk.Label(control_frame, text="\nEXPLANATION:\n🟢 ENTHALTEN: Voice is in clone\n🟡 UNSICHER: Similar but not confirmed\n🔴 NICHT ENTHALTEN: Not in clone", 
                      font=("Arial", 9), justify=tk.LEFT, relief=tk.GROOVE, padx=10, pady=10)
info_label.pack(fill=tk.X, pady=10)

status_label = tk.Label(control_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W, padx=5)
status_label.pack(fill=tk.X, side=tk.BOTTOM, ipady=2, pady=(10, 0))

result_frame = tk.Frame(main_frame)
result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

scrollbar = tk.Scrollbar(result_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

result_text = tk.Text(result_frame, height=30, state="disabled", font=("Courier", 10), yscrollcommand=scrollbar.set)
result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.config(command=result_text.yview)

root.mainloop()