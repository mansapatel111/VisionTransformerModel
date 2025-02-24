
This Vision Transformer implementation adapts the Transformer architecture — originally designed for sequences — to process 2D, 3D and 4D images. 
🚀 Summary of the project:

- Developed and optimized a Vision Transformer (ViT) for autoregressive weather forecasting, leveraging techniques from state-of-the-art AI models (GraphCast, Pangu-Weather, GenCast).
- Implemented autoregressive modeling, scheduled sampling, and multi-scale loss functions to enhance predictive accuracy and reduce long-term error accumulation.
- Processed real-world coastal simulation data (2D spatial-temporal datasets) and designed an efficient train-test pipeline with PyTorch DataLoaders.
- Designed custom evaluation metrics (RMSE, MAE, Pearson Correlation) to quantitatively assess model performance.
- Proposed future extensions for 3D and 4D ViT models to expand forecasting capabilities to volumetric and spatiotemporal datasets.
- Planned and developed model visualization techniques, enabling comparison of model predictions against real-world ground truth data.


Set Up:
* Begin with cloning this repo and then navingating to your repository. Start with using cd my_repository in your terminal to get to the appropriate folder.

* You must create a virtual environment and download all the dependencies. Use the following commands for this: Create a virtual environment: python3 -m venv myenv Running the virtual environment for mac: source myenv/bin/activate Running the virtual environment for windows: .\myenv\Scripts\activate.ps1
  
* Once you see (myenv) at the beginning of your terminal command, you are ready for set up. Start with installing pytorch and any other dependencies not already in your system
  
* Finally, Run the main file to test the code!
