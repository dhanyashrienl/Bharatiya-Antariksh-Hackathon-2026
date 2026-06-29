# Bharatiya-Antariksh-Hackathon-2026
PROBLEM STATEMENT 10 - IR Satellite Image Super-Resolution & Colorization Pipeline

## INTRODUCTION

Infrared (IR) satellite images are crucial for night-time and all-weather observation.
However, IR images are:
      Low in spatial resolution
      Monochrome in nature
      Difficult to interpret visually.
This limits their effectiveness in applications such as urban monitoring, disaster management, and defense surveillance.

## PROPOSED SOLUTION

This project presents a unified deep learning pipeline that:
      Enhances infrared satellite images using super-resolution
      Converts enhanced IR images into realistic RGB colorized outputs
The solution is optimized for Indian satellite data and is suitable for operational deployment.

## KEY CONTRIBUTIONS
      End-to-end two-stage pipeline (SR + Colorization).
      Trained on Landsat-9 Indian subcontinent data.
      Improves visual interpretability of IR images.
      Designed to run on consumer-grade GPU hardware.
      
## METHODOLOGY

Input infrared satellite image.
Super-resolution network enhances spatial details.
Colorization network predicts RGB information.
Final high-resolution(Pesudo) colorized output is generated.

## DATASET

Source: Landsat 9 (Thermal Infrared & RGB bands).
Region: Indian Subcontinent.
Processing: Image tiling, normalization, alignment.

## TECHNOLOGIES USED 

Language: Python.
Framework: PyTorch.
Libraries: NumPy, OpenCV, Matplotlib.
Platform: Kaggle.
Version Control: GitHub.
