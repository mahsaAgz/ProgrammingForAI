# programmingForAi
This is course work for Programming for AI 

# 1.GAN for Face Generation using CelebA

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
