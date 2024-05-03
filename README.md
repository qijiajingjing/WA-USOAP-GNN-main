# Weak-Attention and Unified Smooth Overlap of Atomic Positions Descriptors Graph Neural Network (WA-USOAP-GNN)

## About the Model
In the paper titled "Weak-Attention and Unified Smooth Overlap of Atomic Positions Descriptors Graph Neural Network for Predicting Binding Energies", we introduce a novel network architecture model for graph-based learning. Here are the key features:
- **Unified Smooth Overlap of Atomic Positions (USOAP) Descriptors**: 
These descriptors are used as feature vectors, which, when combined with node features in the graph dataset, enable the model to extract and utilize latent spatial structural information effectively.
- **Weak Attention Mechanism**: 
For the first time in a graph neural network architecture, we implement a weak attention mechanism. This mechanism is applied before the atomic graph convolutional layer and uses a global attention strategy to incorporate global information from the atomic graph, significantly enhancing the model's learning capabilities.

## Installation
- Python 3.x
- PyTorch
- TensorBoardX
- tqdm
- scikit-learn
- NumPy
- dgl
- pymatgen
...
Make sure all dependencies are installed using the following command:
```bash
pip install torch tensorboardX tqdm scikit-learn numpy dgl pymatgen ...
```

## Configuration
The task is configured via a JSON file located in the ‘.config/’ directory, which specifies various settings including model type, model parameters, training procedures, and GPU usage. This configuration allows for the precise tuning of the model architecture, training dynamics, and computational resources, facilitating adaptability to different datasets and optimization for specific performance goals. Users can easily modify this file to experiment with different settings and achieve optimal results for their machine learning tasks.

## Generate data
The '.data/make_row_data.py' script is designed to process crystallographic information files (CIF) and convert them into graph data suitable for use with graph neural network models. This transformation is crucial for modeling materials' properties based on their atomic structures using machine learning techniques.

**Key Features**
- **Conversion of CIF to Graph**: The script reads CIF files and converts them into graph structures capturing the atomic arrangements and bonding.
- **Feature Extraction**: Utilizes SOAP descriptors and other specified features to capture the local atomic environments effectively.
- **Flexibility and Extensibility**: The script can be adapted to various types of CIF files and different properties by adjusting the configuration file and command-line arguments.

**Input Data Format**
- **CIF Files**: Directory specified by --data_path_root containing CIF files.
- **Property File (id_prop.csv)**: A CSV file with columns for material IDs and their properties.
- **Configuration File (config.json)**: Specifies how to extract node features and other graph construction parameters.

**Output Data Format**
- **Graph Data**: Consists of nodes, edges, and node features converted into graph format suitable for graph neural networks.
- **Serialized Data File**: Contains the graph data, including node features, edge connections, and properties, serialized into a file specified by --save_path.

## Usage
You can access the pre-prepared dataset in the '.data/molecules/dataset/' directory (where we have placed a dataset containing oxygen atoms), which can be used to train the model.

You can train the model based on this information using the following command
```bash
python main_catalyze_graph_regression.py
```

## Additional Notes
- Ensure that the CUDA environment is correctly set up if you are using a GPU.
- The script uses K-fold cross-validation for assessing the model performance.
- Adjust the epochs, batch_size, and other parameters as needed based on your dataset and GPU capabilities.
