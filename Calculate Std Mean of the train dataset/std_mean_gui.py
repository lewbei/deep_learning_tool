#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar

def get_mean_std(loader, progress_var):
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0
    total_batches = len(loader)

    for i, (data, _) in enumerate(loader):
        # Sum over batch, height, and width dimensions
        channels_sum += torch.sum(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.sum(data**2, dim=[0, 2, 3])
        num_batches += data.size(0)
        
        # Update progress bar
        progress_var.set((i + 1) / total_batches * 100)
        root.update_idletasks()

    # Calculate mean and std
    num_pixels = num_batches * data.size(2) * data.size(3)
    mean = channels_sum / num_pixels
    std = (channels_sqrd_sum / num_pixels - mean**2) ** 0.5

    return mean, std

def calculate_and_save_statistics(file_path, size=512, batch_size=64):
    try:
        train_dataset = os.path.join(file_path, "train")
        
        # Check if the main directory exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Main directory not found: {file_path}")
        
        # Check if the train dataset directory exists
        if not os.path.exists(train_dataset):
            raise FileNotFoundError(f"Train dataset directory not found: {train_dataset}")

        # Check if the train dataset directory contains class subdirectories
        class_folders = [d for d in os.listdir(train_dataset) if os.path.isdir(os.path.join(train_dataset, d))]
        if not class_folders:
            raise FileNotFoundError(f"Couldn't find any class folder in {train_dataset}.")

        dataset_name = os.path.basename(file_path)  # This will get the dataset directory name

        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])
        train_set = datasets.ImageFolder(train_dataset, transform=transform)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

        # Calculate mean and std
        with torch.no_grad():
            mean, std = get_mean_std(train_loader, progress_var)

        data = {
            'Dataset Name': [dataset_name],
            'Size': [(size, size)],
            'Mean': [mean.tolist()],  # Convert tensors to lists
            'Standard Deviation': [std.tolist()]
        }
        df = pd.DataFrame(data)

        # Save the Excel file in the selected dataset directory
        excel_path = os.path.join(file_path, 'dataset_statistics.xlsx')

        # Check if the Excel file exists
        if os.path.exists(excel_path):
            # Read the existing Excel file
            existing_data = pd.read_excel(excel_path)
            # Append new data
            updated_data = pd.concat([existing_data, df], ignore_index=True)
        else:
            # Use new data if file does not exist
            updated_data = df

        # Save to Excel
        updated_data.to_excel(excel_path, index=False)

        return mean, std, excel_path

    except FileNotFoundError as e:
        messagebox.showerror("File Not Found", str(e))
    except Exception as e:
        messagebox.showerror("Error", str(e))

def browse_directory():
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        entry_file_path.delete(0, tk.END)
        entry_file_path.insert(0, folder_selected)
        load_image_paths(folder_selected)
        display_image_preview()

def run_calculation():
    file_path = entry_file_path.get()
    size = int(entry_size.get())
    batch_size = int(entry_batch_size.get())
    progress_var.set(0)
    try:
        mean, std, excel_path = calculate_and_save_statistics(file_path, size, batch_size)
        if mean is not None and std is not None:
            label_mean.config(text=f"Mean: {mean}")
            label_std.config(text=f"Std: {std}")
            label_excel_path.config(text=f"Saved to: {excel_path}")
            messagebox.showinfo("Success", "Calculation completed successfully!")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def load_image_paths(folder_path):
    global image_paths, current_image_index
    image_paths = []
    current_image_index = 0
    train_dataset = os.path.join(folder_path, "train")
    if os.path.exists(train_dataset):
        class_folders = [d for d in os.listdir(train_dataset) if os.path.isdir(os.path.join(train_dataset, d))]
        for class_folder in class_folders:
            class_folder_path = os.path.join(train_dataset, class_folder)
            image_files = [os.path.join(class_folder_path, f) for f in os.listdir(class_folder_path) if os.path.isfile(os.path.join(class_folder_path, f))]
            image_paths.extend(image_files)

def display_image_preview():
    global current_image_index
    if image_paths:
        img_path = image_paths[current_image_index]
        img = Image.open(img_path)
        img = img.resize((150, 150), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img

def show_next_image():
    global current_image_index
    if image_paths:
        current_image_index = (current_image_index + 1) % len(image_paths)
        display_image_preview()

def show_previous_image():
    global current_image_index
    if image_paths:
        current_image_index = (current_image_index - 1) % len(image_paths)
        display_image_preview()

# Create the main application window
root = tk.Tk()
root.title("Dataset Statistics Calculator")

# Create and place the input fields and labels
tk.Label(root, text="Dataset Directory:").grid(row=0, column=0, padx=10, pady=10)
entry_file_path = tk.Entry(root, width=50)
entry_file_path.grid(row=0, column=1, padx=10, pady=10)
tk.Button(root, text="Browse", command=browse_directory).grid(row=0, column=2, padx=10, pady=10)

tk.Label(root, text="Image Size:").grid(row=1, column=0, padx=10, pady=10)
entry_size = tk.Entry(root, width=10)
entry_size.grid(row=1, column=1, padx=10, pady=10)
entry_size.insert(0, "512")

tk.Label(root, text="Batch Size:").grid(row=2, column=0, padx=10, pady=10)
entry_batch_size = tk.Entry(root, width=10)
entry_batch_size.grid(row=2, column=1, padx=10, pady=10)
entry_batch_size.insert(0, "64")

# Create and place the progress bar
progress_var = tk.DoubleVar()
progress_bar = Progressbar(root, variable=progress_var, maximum=100)
progress_bar.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

# Create and place the run button
tk.Button(root, text="Calculate", command=run_calculation).grid(row=4, column=0, columnspan=3, padx=10, pady=20)

# Create and place the labels for displaying results
label_mean = tk.Label(root, text="Mean: ")
label_mean.grid(row=5, column=0, columnspan=3, padx=10, pady=10)

label_std = tk.Label(root, text="Std: ")
label_std.grid(row=6, column=0, columnspan=3, padx=10, pady=10)

label_excel_path = tk.Label(root, text="Saved to: ")
label_excel_path.grid(row=7, column=0, columnspan=3, padx=10, pady=10)

# Create and place the image preview panel
panel = tk.Label(root)
panel.grid(row=8, column=1, padx=10, pady=10)

# Create and place the navigation buttons
tk.Button(root, text="<<", command=show_previous_image).grid(row=8, column=0, padx=10, pady=10)
tk.Button(root, text=">>", command=show_next_image).grid(row=8, column=2, padx=10, pady=10)

# Initialize global variables
image_paths = []
current_image_index = 0

# Run the application
root.mainloop()


# In[ ]:




