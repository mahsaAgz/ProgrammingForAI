# programmingForAi
This is course work for Programming for AI 

# Project 1: GAN for Face Generation using CelebA

This project implements a **Generative Adversarial Network (GAN)** to generate realistic human face images using the **CelebA dataset**. The model consists of a **Generator** and **Discriminator**, both implemented in **PyTorch**, trained on a subset of **10,000 images**.  

## **Project Overview**  
- **Dataset**: CelebA (10,000 training images, 1,000 for evaluation)  
- **Model Architecture**:  
  - **Generator**: Transforms random noise into a 64×64 RGB image using fully connected layers.  
  - **Discriminator**: A convolutional network that classifies images as real or fake.  
- **Training Setup**:  
  - **Loss Function**: Binary Cross Entropy (BCE)  
  - **Optimizer**: AdamW (learning rate: 0.0001)  
  - **Training**: Runs for 5 epochs with an option to stop at a specified epoch.  
- **Reproducibility**: A fixed random seed ensures consistent training results.  

## **Usage**  
Run the script with the following command:  
```bash
python celebA_GAN.py --seed 0 --img_path ./celeba --stop_epoch 10 --output_path ./
```

# **Project 2: Custom Language Model (Transformer & RNN) for ELI5 Dataset**

This project implements a **custom language model** that combines **Transformer decoder layers** and a **Recurrent Neural Network (RNN)** to train on the **ELI5 dataset**.

## **Project Overview**  
- **Architecture**:
  - Transformer **Decoder Layer (3 layers)**
  - **RNN** (3 layers, hidden size: 64)
  - **Fully Connected Layer**
  - **Another Transformer Decoder Layer (3 layers)**
- **Dataset**:  
  - **ELI5 dataset** (Reddit Q&A)
  - **Training samples**: 14,790, **Validation**: 5,655
  - **Sequence length**: 200 tokens  
- **Training Setup**:
  - **Loss**: Cross-Entropy
  - **Optimizer**: Adam (LR: 5e-4)
  - **Batch Size**: 32
  - **Epochs**: Up to 20 (stops at specified epoch)

## **Usage**  
Run the script with:
```bash
python eli_Lm.py --seed 1 --stop_epoch 2 --output_path ./

