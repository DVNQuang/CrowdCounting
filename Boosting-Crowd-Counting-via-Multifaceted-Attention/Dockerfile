FROM python:3.8

RUN apt-get update && apt-get install -y libgl1-mesa-glx
# Use a base image, for example, the official Python image
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

# Install libGL libraries

RUN pip install scipy
RUN pip install opencv-python
# Copy your code and other dependencies
