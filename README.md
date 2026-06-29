# Bharatiya-Antariksh-Hackathon-2026
PROBLEM STATEMENT 10 - IR Satellite Image Super-Resolution & Colorization Pipeline

## INTRODUCTION

Infrared (IR) satellite images are crucial for night-time and all-weather observation.
However, IR images are:
      Low in spatial resolution.<br>
      Monochrome in nature.<br>
      Difficult to interpret visually.<br>
This limits their effectiveness in applications such as urban monitoring, disaster management, and defense surveillance.

## PROPOSED SOLUTION

This project presents a unified deep learning pipeline that:<br>
      Enhances infrared satellite images using super-resolution.<br>
      Converts enhanced IR images into realistic RGB colorized outputs.<br>
The solution is optimized for Indian satellite data and is suitable for operational deployment.

## KEY CONTRIBUTIONS
      End-to-end two-stage pipeline (SR + Colorization).
      Trained on Landsat-9 Indian subcontinent data.
      Improves visual interpretability of IR images.
      Designed to run on consumer-grade GPU hardware.
      
## METHODOLOGY

Input infrared satellite image.<br>
Super-resolution network enhances spatial details.<br>
Colorization network predicts RGB information.<br>
Final high-resolution(Pesudo) colorized output is generated.

## DATASET

Source: Landsat 9 (Thermal Infrared & RGB bands).<br>
Region: Indian Subcontinent.<br>
Processing: Image tiling, normalization, alignment.

## TECHNOLOGIES USED 

Language: Python.<br>
Framework: PyTorch.<br>
Libraries: NumPy, OpenCV, Matplotlib.<br>
Platform: Kaggle.<br>
Version Control: GitHub.<br>
